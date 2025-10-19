import os
import io
import gc
import time
import base64
import json
import threading
from typing import Optional, Tuple

import torch
import yaml
from PIL import Image
import cairosvg
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoProcessor

from decoder import SketchDecoder
from tokenizer import SVGTokenizer
from qwen_vl_utils import process_vision_info

# -----------------------
# 全局状态（仅初始化一次）
# -----------------------
app = FastAPI(title="SVG Generator EAS Service", version="1.0.0")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = None
processor = None
sketch_decoder = None
svg_tokenizer = None
config = None

SYSTEM_PROMPT = "You are a multimodal SVG generation assistant capable of generating SVG code from both text descriptions and images."
SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']

# 线程锁，避免多并发导致显存竞争
gen_lock = threading.Lock()


# -----------------------
# 工具函数
# -----------------------
def b64_to_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def pil_to_b64(img: Image.Image, format="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def process_and_resize_image(image_input, target_size=(200, 200)) -> Image.Image:
    if isinstance(image_input, str):
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.fromarray(image_input)
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return image


def process_text_to_svg_inputs(text_description: str):
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Task: text-to-svg\nDescription: {text_description}\nGenerate SVG code based on the above description."}
        ]
    }]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_input],
        truncation=True,
        return_tensors="pt"
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    pixel_values = None
    image_grid_thw = None
    return input_ids, attention_mask, pixel_values, image_grid_thw


def process_image_to_svg_inputs(image_pil: Image.Image):
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": [
            {"type": "text", "text": "Task: image-to-svg\nGenerate SVG code that accurately represents the following image."},
            {"type": "image", "image": image_pil},
        ]
    }]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    pixel_values = inputs.get('pixel_values', None)
    image_grid_thw = inputs.get('image_grid_thw', None)

    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)
    return input_ids, attention_mask, pixel_values, image_grid_thw


def generate_svg(input_ids, attention_mask, pixel_values=None, image_grid_thw=None, task_type="image-to-svg") -> Tuple[str, Optional[Image.Image]]:
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if task_type == "image-to-svg":
            gen_config = dict(
                do_sample=True,
                temperature=0.1,
                top_p=0.001,
                top_k=1,
                repetition_penalty=1.05,
                early_stopping=True,
            )
        else:
            gen_config = dict(
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.05,
                early_stopping=True,
            )

        model_config = config['model']
        max_length = model_config['max_length']
        output_ids = torch.ones(1, max_length + 1).long().to(device) * model_config['eos_token_id']

        with torch.no_grad():
            results = sketch_decoder.transformer.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_length,
                num_return_sequences=1,
                bos_token_id=model_config['bos_token_id'],
                eos_token_id=model_config['eos_token_id'],
                pad_token_id=model_config['pad_token_id'],
                use_cache=True,
                **gen_config
            )

            results = results[:, :max_length]
            output_ids[:, :results.shape[1]] = results

            generated_xy, generated_colors = svg_tokenizer.process_generated_tokens(output_ids)

        svg_tensors = svg_tokenizer.raster_svg(generated_xy)
        if not svg_tensors or not svg_tensors[0]:
            return "Error: No valid SVG paths generated", None

        svg = svg_tokenizer.apply_colors_to_svg(svg_tensors[0], generated_colors)
        svg_str = svg.to_str()

        png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
        png_image = Image.open(io.BytesIO(png_data))

        return svg_str, png_image

    except Exception as e:
        return f"Error: {str(e)}", None


def load_models_once():
    global tokenizer, processor, sketch_decoder, svg_tokenizer, config

    if all([tokenizer, processor, sketch_decoder, svg_tokenizer, config]):
        return

    # 环境变量
    weight_path = os.environ.get("WEIGHT_PATH", "/mnt/data/OmniSVG")
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    qwen_dir = os.environ.get("QWEN_LOCAL_DIR", "/mnt/data/Qwen2.5-VL-3B-Instruct")
    svg_tokenizer_config = os.environ.get("SVG_TOKENIZER_CONFIG", config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"CONFIG_PATH not found: {config_path}")
    if not os.path.isdir(weight_path):
        raise FileNotFoundError(f"WEIGHT_PATH not found: {weight_path}")
    if not os.path.isdir(qwen_dir):
        # 如需联网加载，可改成模型名字符串；建议离线部署
        raise FileNotFoundError(f"QWEN_LOCAL_DIR not found: {qwen_dir}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Qwen Processor/Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(qwen_dir, padding_side="left")
    processor = AutoProcessor.from_pretrained(qwen_dir, padding_side="left")

    # 自定义解码器与权重
    sketch_decoder = SketchDecoder()
    weight_file = os.path.join(weight_path, "pytorch_model.bin")
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"weight file not found: {weight_file}")

    state = torch.load(weight_file, map_location="cpu")
    sketch_decoder.load_state_dict(state)
    sketch_decoder.to(device).eval()

    # SVG Tokenizer
    svg_tokenizer = SVGTokenizer(svg_tokenizer_config)


# -----------------------
# 请求/响应模型
# -----------------------
class PredictRequest(BaseModel):
    task_type: str  # "image-to-svg" | "text-to-svg"
    text: Optional[str] = None            # text-to-svg 时必填
    image_base64: Optional[str] = None    # image-to-svg 时可用
    return_png: Optional[bool] = True     # 是否返回PNG预览（base64）


# -----------------------
# 路由
# -----------------------
@app.on_event("startup")
def startup_event():
    # 禁用 tokenizer 并行告警
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    load_models_once()


@app.get("/ping")
def ping():
    try:
        load_models_once()
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


@app.post("/predict")
async def predict_endpoint(req: Request):
    """
    支持两种输入：
    1) application/json:
       {
         "task_type": "image-to-svg" | "text-to-svg",
         "text": "...",                # text-to-svg
         "image_base64": "...",        # image-to-svg
         "return_png": true
       }
    2) multipart/form-data:
       - task_type: same as above
       - text: optional
       - image: UploadFile (二选一 image or image_base64)
       - return_png: optional
    """
    t0 = time.time()
    load_models_once()

    content_type = req.headers.get("content-type", "")
    try:
        if "application/json" in content_type:
            body = await req.json()
            data = PredictRequest(**body)
            image_file = None
        else:
            # 兼容 multipart/form-data
            form = await req.form()
            data = PredictRequest(
                task_type=form.get("task_type", ""),
                text=form.get("text"),
                image_base64=form.get("image_base64"),
                return_png=(form.get("return_png", "true").lower() == "true")
            )
            image_file = form.get("image")  # type: ignore

        if data.task_type not in ("image-to-svg", "text-to-svg"):
            return JSONResponse(status_code=400, content={"error": "task_type must be 'image-to-svg' or 'text-to-svg'"})

        with gen_lock:
            if data.task_type == "text-to-svg":
                if not data.text:
                    return JSONResponse(status_code=400, content={"error": "text is required for text-to-svg"})
                input_ids, attention_mask, pixel_values, image_grid_thw = process_text_to_svg_inputs(data.text)
                svg_code, png_image = generate_svg(input_ids, attention_mask, pixel_values, image_grid_thw, "text-to-svg")

            else:
                # image-to-svg
                image_pil = None
                if data.image_base64:
                    image_pil = b64_to_pil(data.image_base64)
                elif image_file and isinstance(image_file, UploadFile):
                    raw = await image_file.read()
                    image_pil = Image.open(io.BytesIO(raw)).convert("RGBA")
                else:
                    return JSONResponse(status_code=400, content={"error": "image_base64 or multipart image is required for image-to-svg"})

                processed = process_and_resize_image(image_pil)
                input_ids, attention_mask, pixel_values, image_grid_thw = process_image_to_svg_inputs(processed)
                svg_code, png_image = generate_svg(input_ids, attention_mask, pixel_values, image_grid_thw, "image-to-svg")

        if not svg_code or str(svg_code).startswith("Error"):
            return JSONResponse(status_code=500, content={"error": str(svg_code)})

        resp = {
            "svg": svg_code,
            "elapsed_ms": int((time.time() - t0) * 1000)
        }
        if png_image is not None and (data.return_png is None or data.return_png):
            resp["png_base64"] = pil_to_b64(png_image, format="PNG")
        return JSONResponse(content=resp)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    # 本地调试：python service.py
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("service:app", host="0.0.0.0", port=port, workers=1)

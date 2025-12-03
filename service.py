import base64
import gc
import io
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

import cairosvg
import numpy as np
import torch
import yaml
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, AutoTokenizer

from decoder import SketchDecoder
from tokenizer import SVGTokenizer
from qwen_vl_utils import process_vision_info


app = FastAPI(title="OmniSVG Runpod Service", version="2.0.0")

CONFIG_PATH = os.environ.get("CONFIG_PATH", "./config.yaml")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"CONFIG_PATH not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SYSTEM_PROMPT = (
    "You are an expert SVG code generator. "
    "Generate precise, valid SVG path commands that accurately represent the described scene or object. "
    "Focus on capturing key shapes, spatial relationships, and visual composition."
)

AVAILABLE_MODEL_SIZES = list(config.get("models", {}).keys())
DEFAULT_MODEL_SIZE = (
    config.get("default_model_size", AVAILABLE_MODEL_SIZES[0]) if AVAILABLE_MODEL_SIZES else None
)

image_config = config.get("image", {})
TARGET_IMAGE_SIZE = image_config.get("target_size", 448)
RENDER_SIZE = image_config.get("render_size", 512)
BACKGROUND_THRESHOLD = image_config.get("background_threshold", 240)
EMPTY_THRESHOLD_ILLUSTRATION = image_config.get("empty_threshold_illustration", 250)
EMPTY_THRESHOLD_ICON = image_config.get("empty_threshold_icon", 252)
EDGE_SAMPLE_RATIO = image_config.get("edge_sample_ratio", 0.1)
COLOR_SIMILARITY_THRESHOLD = image_config.get("color_similarity_threshold", 30)
MIN_EDGE_SAMPLES = image_config.get("min_edge_samples", 10)

colors_config = config.get("colors", {})
BLACK_COLOR_TOKEN = colors_config.get(
    "black_color_token", colors_config.get("color_token_start", 40010) + 2
)

model_config = config.get("model", {})
BOS_TOKEN_ID = model_config.get("bos_token_id", 196998)
EOS_TOKEN_ID = model_config.get("eos_token_id", 196999)
PAD_TOKEN_ID = model_config.get("pad_token_id", 151643)

MAX_LENGTH_MIN = 256
MAX_LENGTH_MAX = 2048
MAX_LENGTH_DEFAULT = 512

task_config = config.get("task_configs", {})
TASK_CONFIGS = {
    "text-to-svg-icon": task_config.get(
        "text_to_svg_icon",
        {
            "default_temperature": 0.5,
            "default_top_p": 0.88,
            "default_top_k": 50,
            "default_repetition_penalty": 1.05,
        },
    ),
    "text-to-svg-illustration": task_config.get(
        "text_to_svg_illustration",
        {
            "default_temperature": 0.6,
            "default_top_p": 0.90,
            "default_top_k": 60,
            "default_repetition_penalty": 1.03,
        },
    ),
    "image-to-svg": task_config.get(
        "image_to_svg",
        {
            "default_temperature": 0.3,
            "default_top_p": 0.90,
            "default_top_k": 50,
            "default_repetition_penalty": 1.05,
        },
    ),
}

gen_config = config.get("generation", {})
DEFAULT_NUM_CANDIDATES = max(1, gen_config.get("default_num_candidates", 1))
MAX_NUM_CANDIDATES = max(DEFAULT_NUM_CANDIDATES, gen_config.get("max_num_candidates", 4))
EXTRA_CANDIDATES_BUFFER = max(0, gen_config.get("extra_candidates_buffer", 2))

validation_config = config.get("validation", {})
MIN_SVG_LENGTH = validation_config.get("min_svg_length", 20)

SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

tokenizer: Optional[AutoTokenizer] = None
processor: Optional[AutoProcessor] = None
sketch_decoder: Optional[SketchDecoder] = None
svg_tokenizer: Optional[SVGTokenizer] = None
current_model_size: Optional[str] = None

model_loading_lock = threading.Lock()
gen_lock = threading.Lock()


def get_config_value(model_size: str, *keys: str) -> Optional[Any]:
    value: Any = config.get("models", {}).get(model_size, {})
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            value = None
            break
    if value is not None:
        return value
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def is_local_path(path: str) -> bool:
    if os.path.exists(path):
        return True
    if path.startswith(("./", "../", "/")):
        return True
    if os.path.sep in path and os.path.exists(os.path.dirname(path)):
        return True
    if len(path) > 1 and path[1] == ":":
        return True
    return False


def resolve_weight_source(model_size: str) -> Optional[str]:
    env_key = f"WEIGHT_PATH_{model_size.upper()}"
    path = os.environ.get(env_key)
    if path:
        return path
    if os.environ.get("WEIGHT_PATH"):
        return os.environ["WEIGHT_PATH"]
    return get_config_value(model_size, "huggingface", "omnisvg_model")


def resolve_model_source(model_size: str) -> Optional[str]:
    env_key = f"QWEN_MODEL_{model_size.upper()}"
    path = os.environ.get(env_key)
    if path:
        return path
    if os.environ.get("QWEN_LOCAL_DIR"):
        return os.environ["QWEN_LOCAL_DIR"]
    return get_config_value(model_size, "huggingface", "qwen_model")


def download_model_weights(repo_or_path: str, filename: str = "pytorch_model.bin") -> str:
    if is_local_path(repo_or_path):
        if repo_or_path.endswith(".bin"):
            return repo_or_path
        file_path = os.path.join(repo_or_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{filename} not found under {repo_or_path}")
        return file_path
    return hf_hub_download(repo_id=repo_or_path, filename=filename, resume_download=True)


def load_models(model_size: str) -> None:
    global tokenizer, processor, sketch_decoder, svg_tokenizer, current_model_size

    model_path = resolve_model_source(model_size)
    weight_path = resolve_weight_source(model_size)
    if not model_path or not weight_path:
        raise ValueError(f"Missing paths for model size {model_size}")

    print(f"\n{'=' * 60}\nLoading {model_size} model\n{'=' * 60}")
    print(f"Qwen model: {model_path}")
    print(f"OmniSVG weights: {weight_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    sketch_decoder = SketchDecoder(
        config_path=CONFIG_PATH,
        model_path=model_path,
        model_size=model_size,
        pix_len=MAX_LENGTH_MAX,
        text_len=config.get("text", {}).get("max_length", 200),
        torch_dtype=DTYPE,
    )

    weight_file = download_model_weights(weight_path)
    state_dict = torch.load(weight_file, map_location="cpu")
    sketch_decoder.load_state_dict(state_dict)
    sketch_decoder = sketch_decoder.to(device).eval()

    svg_tokenizer = SVGTokenizer(CONFIG_PATH, model_size=model_size)
    current_model_size = model_size
    print(f"Loaded {model_size} model successfully!")


def ensure_model_loaded(model_size: Optional[str] = None) -> None:
    global current_model_size, tokenizer, processor, sketch_decoder, svg_tokenizer
    target_size = model_size or DEFAULT_MODEL_SIZE
    if target_size is None:
        raise ValueError("No model_size specified and default is unavailable")
    if current_model_size == target_size and sketch_decoder is not None:
        return
    with model_loading_lock:
        if current_model_size == target_size and sketch_decoder is not None:
            return
        if current_model_size is not None:
            print(f"Switching model: {current_model_size} -> {target_size}")
            sketch_decoder = None
            tokenizer = None
            processor = None
            svg_tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        load_models(target_size)


def b64_to_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def detect_text_subtype(text_prompt: str) -> str:
    text_lower = text_prompt.lower()
    icon_keywords = [
        "icon",
        "logo",
        "symbol",
        "badge",
        "button",
        "emoji",
        "glyph",
        "simple",
        "arrow",
        "triangle",
        "circle",
        "square",
        "heart",
        "star",
        "checkmark",
    ]
    if any(kw in text_lower for kw in icon_keywords):
        return "icon"
    illustration_keywords = [
        "illustration",
        "scene",
        "person",
        "people",
        "character",
        "animal",
        "cat",
        "dog",
        "bird",
        "landscape",
        "city",
        "ocean",
        "sunset",
        "forest",
        "mountain",
    ]
    if any(kw in text_lower for kw in illustration_keywords) or len(text_prompt) > 50:
        return "illustration"
    return "icon"


def detect_and_replace_background(
    image: Image.Image,
    threshold: Optional[int] = None,
    edge_sample_ratio: Optional[float] = None,
):
    if threshold is None:
        threshold = BACKGROUND_THRESHOLD
    if edge_sample_ratio is None:
        edge_sample_ratio = EDGE_SAMPLE_RATIO

    img_array = np.array(image)
    if image.mode == "RGBA":
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(bg, image)
        return composite.convert("RGB"), True

    h, w = img_array.shape[:2]
    edge_pixels = []
    sample_count = max(MIN_EDGE_SAMPLES, int(min(h, w) * edge_sample_ratio))

    for i in range(0, w, max(1, w // sample_count)):
        edge_pixels.append(img_array[0, i])
        edge_pixels.append(img_array[h - 1, i])
    for i in range(0, h, max(1, h // sample_count)):
        edge_pixels.append(img_array[i, 0])
        edge_pixels.append(img_array[i, w - 1])

    if edge_pixels:
        mean_edge = np.array(edge_pixels).mean(axis=0)
        if np.all(mean_edge > threshold):
            return image, False

    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        from collections import Counter

        edge_colors = []
        for i in range(w):
            edge_colors.append(tuple(img_array[0, i, :3]))
            edge_colors.append(tuple(img_array[h - 1, i, :3]))
        for i in range(h):
            edge_colors.append(tuple(img_array[i, 0, :3]))
            edge_colors.append(tuple(img_array[i, w - 1, :3]))
        color_counts = Counter(edge_colors)
        bg_color = color_counts.most_common(1)[0][0]
        color_diff = np.sqrt(
            np.sum((img_array[:, :, :3].astype(float) - np.array(bg_color)) ** 2, axis=2)
        )
        bg_mask = color_diff < COLOR_SIMILARITY_THRESHOLD
        result = img_array.copy()
        if result.shape[2] == 4:
            result[bg_mask] = [255, 255, 255, 255]
        else:
            result[bg_mask] = [255, 255, 255]
        return Image.fromarray(result).convert("RGB"), True
    return image, False


def preprocess_image_for_svg(
    image: Image.Image, replace_background: bool = True, target_size: Optional[int] = None
):
    if target_size is None:
        target_size = TARGET_IMAGE_SIZE
    if image.mode == "RGBA":
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        img_with_bg = Image.alpha_composite(bg, image).convert("RGB")
        was_modified = True
    elif image.mode in ("LA", "PA"):
        image = image.convert("RGBA")
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        img_with_bg = Image.alpha_composite(bg, image).convert("RGB")
        was_modified = True
    elif image.mode != "RGB":
        img_with_bg = image.convert("RGB")
        was_modified = True
    else:
        img_with_bg = image
        was_modified = False
    if replace_background:
        img_with_bg, bg_replaced = detect_and_replace_background(img_with_bg)
        was_modified = was_modified or bg_replaced
    img_resized = img_with_bg.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return img_resized, was_modified


def prepare_inputs(task_type: str, content: Any):
    if task_type == "text-to-svg":
        instruction = (
            f"Generate an SVG illustration for: {content}\n"
            "Requirements:\n"
            "- Create complete SVG path commands\n"
            "- Include proper coordinates and colors\n"
            "- Maintain visual clarity and composition"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": instruction}]},
        ]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(text=[text_input], padding=True, truncation=True, return_tensors="pt")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate SVG code that accurately represents this image:"},
                {"type": "image", "image": content},
            ],
        },
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return processor(
        text=[text_input], images=image_inputs, padding=True, truncation=True, return_tensors="pt"
    )


def render_svg_to_image(svg_str: str, size: Optional[int] = None) -> Optional[Image.Image]:
    size = size or RENDER_SIZE
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"), output_width=size, output_height=size
        )
        image_rgba = Image.open(io.BytesIO(png_data)).convert("RGBA")
        bg = Image.new("RGB", image_rgba.size, (255, 255, 255))
        bg.paste(image_rgba, mask=image_rgba.split()[3])
        return bg
    except Exception as exc:
        print(f"Render error: {exc}")
        return None


def is_valid_candidate(svg_str: str, img: Optional[Image.Image], subtype: str) -> bool:
    if not svg_str or len(svg_str) < MIN_SVG_LENGTH:
        return False
    if "<svg" not in svg_str:
        return False
    if img is None:
        return False
    mean_val = np.array(img).mean()
    threshold = EMPTY_THRESHOLD_ILLUSTRATION if subtype == "illustration" else EMPTY_THRESHOLD_ICON
    return mean_val <= threshold


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_params(task_type: str, subtype: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if task_type == "text-to-svg" and subtype == "icon":
        defaults = TASK_CONFIGS["text-to-svg-icon"]
    elif task_type == "text-to-svg":
        defaults = TASK_CONFIGS["text-to-svg-illustration"]
    else:
        defaults = TASK_CONFIGS["image-to-svg"]

    def pick(name: str, fallback: float) -> float:
        key = f"default_{name}"
        value = params.get(name)
        if value is None:
            value = defaults.get(key, fallback)
        return value

    temperature = float(pick("temperature", 0.5))
    top_p = float(pick("top_p", 0.9))
    top_k = int(pick("top_k", 50))
    repetition_penalty = float(pick("repetition_penalty", 1.05))
    num_candidates = int(params.get("num_candidates", DEFAULT_NUM_CANDIDATES))
    max_length = int(params.get("max_length", MAX_LENGTH_DEFAULT))

    return {
        "temperature": _clamp(temperature, 0.05, 1.5),
        "top_p": _clamp(top_p, 0.1, 1.0),
        "top_k": int(_clamp(top_k, 1, 200)),
        "repetition_penalty": _clamp(repetition_penalty, 1.0, 2.0),
        "num_candidates": max(1, min(num_candidates, MAX_NUM_CANDIDATES)),
        "max_length": max(MAX_LENGTH_MIN, min(max_length, MAX_LENGTH_MAX)),
    }


def generate_candidates(
    inputs,
    task_type: str,
    subtype: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_length: int,
    num_samples: int,
) -> List[Dict[str, Any]]:
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    model_inputs: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "pixel_values" in inputs:
        model_inputs["pixel_values"] = inputs["pixel_values"].to(device, dtype=DTYPE)
    if "image_grid_thw" in inputs:
        model_inputs["image_grid_thw"] = inputs["image_grid_thw"].to(device)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gen_cfg = {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": int(top_k),
        "repetition_penalty": repetition_penalty,
        "early_stopping": True,
        "no_repeat_ngram_size": 0,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "bos_token_id": BOS_TOKEN_ID,
    }
    actual_samples = min(
        MAX_NUM_CANDIDATES + EXTRA_CANDIDATES_BUFFER, num_samples + EXTRA_CANDIDATES_BUFFER
    )
    candidates: List[Dict[str, Any]] = []

    with gen_lock:
        with torch.no_grad():
            results = sketch_decoder.transformer.generate(
                **model_inputs,
                max_new_tokens=int(max_length),
                num_return_sequences=int(actual_samples),
                use_cache=True,
                **gen_cfg,
            )
        input_len = input_ids.shape[1]
        generated_ids_batch = results[:, input_len:]

    for i in range(min(actual_samples, generated_ids_batch.shape[0])):
        current_ids = generated_ids_batch[i : i + 1]
        fake_wrapper = torch.cat(
            [
                torch.full((1, 1), BOS_TOKEN_ID, device=device),
                current_ids,
                torch.full((1, 1), EOS_TOKEN_ID, device=device),
            ],
            dim=1,
        )
        try:
            generated_xy = svg_tokenizer.process_generated_tokens(fake_wrapper)
            if len(generated_xy) == 0:
                continue
            svg_tensors, color_tensors = svg_tokenizer.raster_svg(generated_xy)
            if not svg_tensors or not svg_tensors[0]:
                continue
            num_paths = len(svg_tensors[0])
            while len(color_tensors) < num_paths:
                color_tensors.append(BLACK_COLOR_TOKEN)
            svg = svg_tokenizer.apply_colors_to_svg(svg_tensors[0], color_tensors)
            svg_str = svg.to_str()
            if "width=" not in svg_str:
                svg_str = svg_str.replace(
                    "<svg", f'<svg width="{TARGET_IMAGE_SIZE}" height="{TARGET_IMAGE_SIZE}"', 1
                )
            if "viewBox" not in svg_str:
                svg_str = svg_str.replace(
                    "<svg", f'<svg viewBox="0 0 {TARGET_IMAGE_SIZE} {TARGET_IMAGE_SIZE}"', 1
                )
            png_image = render_svg_to_image(svg_str, size=RENDER_SIZE)
            if not is_valid_candidate(svg_str, png_image, subtype):
                continue
            candidates.append(
                {
                    "index": len(candidates) + 1,
                    "svg": svg_str,
                    "img": png_image,
                    "path_count": num_paths,
                }
            )
            if len(candidates) >= num_samples:
                break
        except Exception as exc:
            print(f"Candidate {i} failed: {exc}")
            continue
    return candidates


class PredictRequest(BaseModel):
    task_type: str
    text: Optional[str] = None
    image_base64: Optional[str] = None
    model_size: Optional[str] = None
    task_subtype: Optional[str] = None
    num_candidates: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_length: Optional[int] = None
    replace_background: Optional[bool] = True
    return_png: Optional[bool] = True


def run_generation(req: PredictRequest, image_override: Optional[Image.Image] = None) -> Dict[str, Any]:
    if req.task_type not in {"text-to-svg", "image-to-svg"}:
        raise ValueError("task_type must be 'text-to-svg' or 'image-to-svg'")

    model_size = (req.model_size or DEFAULT_MODEL_SIZE or "").strip() or DEFAULT_MODEL_SIZE
    if model_size not in AVAILABLE_MODEL_SIZES:
        raise ValueError(f"model_size must be one of {AVAILABLE_MODEL_SIZES}")

    ensure_model_loaded(model_size)
    subtype = req.task_subtype
    inputs = None
    processed_image = None

    if req.task_type == "text-to-svg":
        text = (req.text or "").strip()
        if not text:
            raise ValueError("text is required for text-to-svg")
        subtype = subtype or detect_text_subtype(text)
        inputs = prepare_inputs("text-to-svg", text)
    else:
        replace_bg = req.replace_background if req.replace_background is not None else True
        if image_override is not None:
            source_img = image_override
        elif req.image_base64:
            source_img = b64_to_pil(req.image_base64)
        else:
            raise ValueError("image_base64 or uploaded image is required for image-to-svg")
        processed_image, _ = preprocess_image_for_svg(
            source_img, replace_background=replace_bg, target_size=TARGET_IMAGE_SIZE
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            processed_image.save(tmp_file.name, format="PNG", quality=100)
            temp_path = tmp_file.name
        try:
            inputs = prepare_inputs("image-to-svg", temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        subtype = "image"

    gen_params = normalize_params(req.task_type, subtype or "icon", req.dict())

    start_time = time.time()
    candidates = generate_candidates(
        inputs,
        req.task_type,
        subtype or "icon",
        gen_params["temperature"],
        gen_params["top_p"],
        gen_params["top_k"],
        gen_params["repetition_penalty"],
        gen_params["max_length"],
        gen_params["num_candidates"],
    )
    elapsed = int((time.time() - start_time) * 1000)

    response_candidates: List[Dict[str, Any]] = []
    for cand in candidates:
        entry = {
            "index": cand["index"],
            "svg": cand["svg"],
            "path_count": cand["path_count"],
        }
        if req.return_png:
            entry["png_base64"] = pil_to_b64(cand["img"])
        response_candidates.append(entry)

    status = "ok" if response_candidates else "no_valid_candidates"
    result: Dict[str, Any] = {
        "status": status,
        "task_type": req.task_type,
        "model_size": model_size,
        "subtype": subtype,
        "elapsed_ms": elapsed,
        "parameters": gen_params,
        "num_candidates": len(response_candidates),
        "return_png": bool(req.return_png),
        "candidates": response_candidates,
    }
    if processed_image is not None and req.return_png:
        result["processed_input_png_base64"] = pil_to_b64(processed_image)
    if response_candidates:
        result["primary_svg"] = response_candidates[0]["svg"]
        if req.return_png:
            result["primary_png_base64"] = response_candidates[0].get("png_base64")
    else:
        result["message"] = "No valid SVG generated. Try adjusting parameters."
    return result


def parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


@app.on_event("startup")
def on_startup():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@app.get("/ping")
def ping():
    try:
        result = {"status": "ok", "device": str(device), "loaded_model": current_model_size}
        return result
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(exc)})


@app.post("/predict")
async def predict_endpoint(req: Request):
    t0 = time.time()
    content_type = req.headers.get("content-type", "")
    try:
        image_upload: Optional[UploadFile] = None
        if "application/json" in content_type:
            body = await req.json()
            data = PredictRequest(**body)
        else:
            form = await req.form()
            form_dict: Dict[str, Any] = {k: form.get(k) for k in (
                "task_type",
                "text",
                "image_base64",
                "model_size",
                "task_subtype",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "num_candidates",
                "max_length",
            )}
            bool_fields = {
                "return_png": parse_bool(form.get("return_png")),
                "replace_background": parse_bool(form.get("replace_background")),
            }
            for k, v in bool_fields.items():
                if v is not None:
                    form_dict[k] = v
            for field in ("temperature", "top_p", "repetition_penalty"):
                if form_dict.get(field) is not None:
                    form_dict[field] = float(form_dict[field])
            for field in ("top_k", "num_candidates", "max_length"):
                if form_dict.get(field) is not None:
                    form_dict[field] = int(form_dict[field])
            data = PredictRequest(**form_dict)
            image_upload = form.get("image")  # type: ignore

        image_override = None
        if image_upload and isinstance(image_upload, UploadFile):
            raw = await image_upload.read()
            if raw:
                image_override = Image.open(io.BytesIO(raw)).convert("RGBA")
        result = run_generation(data, image_override=image_override)
        result["request_time_ms"] = int((time.time() - t0) * 1000)
        return JSONResponse(content=result)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


def load_models_once(model_size: Optional[str] = None) -> None:
    ensure_model_loaded(model_size)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("service:app", host="0.0.0.0", port=port, workers=1)

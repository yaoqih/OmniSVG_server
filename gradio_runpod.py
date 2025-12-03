import argparse
import base64
import io
import os
from typing import Any, Dict, List, Optional

import gradio as gr
import yaml
from dotenv import load_dotenv
from PIL import Image

from runpod_client import (
    RunpodClientError,
    encode_image_to_base64,
    ensure_env_ready,
    runsync,
)


load_dotenv()


CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"CONFIG_PATH not found for gradio client: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

image_cfg = _config.get("image", {})
TARGET_IMAGE_SIZE = image_cfg.get("target_size", 448)

AVAILABLE_MODEL_SIZES = list(_config.get("models", {}).keys())
DEFAULT_MODEL_SIZE = _config.get("default_model_size", AVAILABLE_MODEL_SIZES[0] if AVAILABLE_MODEL_SIZES else "8B")

task_config = _config.get("task_configs", {})
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

gen_config = _config.get("generation", {})
DEFAULT_NUM_CANDIDATES = max(1, gen_config.get("default_num_candidates", 1))
MAX_NUM_CANDIDATES = max(DEFAULT_NUM_CANDIDATES, gen_config.get("max_num_candidates", 4))
MAX_LENGTH_MIN = 256
MAX_LENGTH_MAX = 2048
MAX_LENGTH_DEFAULT = 512

CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 24px !important;
    background: #0b1220;
    color: #f3f4f6;
}
.header-container {
    text-align: center;
    margin-bottom: 24px;
    padding: 28px;
    background: linear-gradient(135deg, #111827 0%, #1f1b2e 40%, #2b1345 100%);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 20px 40px rgba(10, 10, 30, 0.65);
}
.header-container h1 {
    margin: 0;
    font-size: 2.5em;
    color: #f8fafc;
    font-weight: 700;
}
.header-container p {
    margin: 12px 0 0 0;
    color: #cbd5f5;
    font-size: 1.05em;
}
.primary-btn {
    background: #05060b !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    font-size: 1.05em !important;
    color: #fefefe !important;
    border-radius: 999px !important;
}
.primary-btn:hover {
    background: #0f172a !important;
}
.settings-group {
    background: #111827;
    border-radius: 18px;
    padding: 20px;
    margin: 14px 0;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 16px 32px rgba(4, 6, 12, 0.65);
}
.code-output textarea {
    font-family: 'JetBrains Mono', 'Monaco', 'Menlo', monospace !important;
    font-size: 12px !important;
    background: #05060b !important;
    color: #f8fafc !important;
    border-radius: 12px !important;
    line-height: 1.4 !important;
}
.input-image {
    border: 2px dashed rgba(255, 255, 255, 0.25);
    border-radius: 16px;
    transition: border-color 0.25s, background 0.25s;
    background: rgba(255, 255, 255, 0.02);
}
.input-image:hover {
    border-color: #f8fafc;
    background: rgba(255, 255, 255, 0.05);
}
.tips-box {
    background: #0f172a;
    border-radius: 20px;
    padding: 26px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
}
.orange-box {
    background: rgba(252, 211, 77, 0.12);
    border-left: 4px solid #fcd34d;
    border-radius: 12px;
    padding: 16px;
    margin: 10px 0;
}
.green-box {
    background: rgba(16, 185, 129, 0.12);
    border-left: 4px solid #34d399;
    border-radius: 12px;
    padding: 16px;
    margin: 10px 0;
}
.blue-box {
    background: rgba(96, 165, 250, 0.12);
    border-left: 4px solid #60a5fa;
    border-radius: 12px;
    padding: 16px;
    margin: 10px 0;
}
.gpu-notice {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 16px 20px;
    margin: 16px 0;
    box-shadow: 0 12px 24px rgba(1, 5, 14, 0.7);
}
"""

TIPS_HTML_BOTTOM = """
<div class="tips-box" style="margin-top: 30px;">
    <h3>💡 Tips & Guide</h3>
    
    <div class="orange-box">
        <strong>🎲 Not getting the result you want?</strong>
        <p style="margin: 8px 0 0 0;">This is normal! <strong>Just click "Generate SVG" again to re-roll.</strong> Each generation is different - try 2-3 times to find the best result!</p>
    </div>
    
    <div class="green-box">
        <strong>📝 Prompting Tips</strong>
        <ul style="margin: 8px 0 0 0; padding-left: 20px;">
            <li><strong>Use geometric descriptions:</strong> "triangular roof", "circular head", "oval body", "curved tail"</li>
            <li><strong>Specify colors for EACH element:</strong> "red roof", "blue shirt", "black outline", "green grass"</li>
            <li><strong>Keep it simple:</strong> Use short, clear phrases connected by commas</li>
            <li><strong>Add positions:</strong> "at top", "in center", "at bottom", "facing right"</li>
        </ul>
    </div>
    
    <div class="blue-box">
        <strong>⚙️ Parameter Guide</strong>
        <ul style="margin: 8px 0 0 0; padding-left: 20px;">
            <li><strong>Max Length:</strong> Lower (256-1024) = faster & simpler | Higher (1024-2048) = slower & more detailed</li>
            <li><strong>Temperature:</strong> Lower (0.2-0.4) = more accurate | Higher (0.5-0.7) = more creative</li>
            <li><strong>Messy result?</strong> Lower temperature and top_k</li>
            <li><strong>Too simple?</strong> Increase max_length and temperature</li>
        </ul>
    </div>
    
    <div style="margin-top: 15px; padding: 14px; background: rgba(30, 64, 175, 0.15); border-radius: 10px; border-left: 4px solid #38bdf8;">
        <strong>✨ Recommended Prompt Structure</strong>
        <div style="background: rgba(15, 23, 42, 0.55); padding: 12px; border-radius: 8px; margin-top: 8px; font-family: monospace; font-size: 0.9em; color: #e0f2fe;">
            [Subject] + [Shape descriptions with colors] + [Position] + [Style]
        </div>
        <p style="margin: 10px 0 0 0; color: #bae6fd; font-size: 0.95em;">
            Example: "A fox logo: triangular orange head, pointed ears, white chest marking, facing right. Minimalist flat style."
        </p>
    </div>
</div>
"""

IMAGE_TIPS_HTML = """
<div class="orange-box">
    <strong>🎲 Tips for Best Results</strong>
    <ul style="margin: 8px 0 0 0; padding-left: 20px;">
        <li><strong>Simple images work best:</strong> Clean backgrounds, clear shapes</li>
        <li><strong>Not satisfied?</strong> Just click generate again to re-roll!</li>
        <li><strong>PNG with transparency</strong> is automatically converted to white background</li>
    </ul>
</div>
"""


def decode_png_base64(b64: Optional[str]) -> Optional[Image.Image]:
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        return None


def find_examples_dir() -> Optional[str]:
    for d in ["examples", "example", "assets/examples", "assets/example"]:
        if os.path.isdir(d):
            return d
    return None


def get_example_texts() -> List[List[str]]:
    ex_dir = find_examples_dir()
    texts: List[List[str]] = []
    if ex_dir:
        for fname in ("texts.txt", "prompts.txt"):
            path = os.path.join(ex_dir, fname)
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            s = line.strip()
                            if s:
                                texts.append([s])
                    break
                except Exception:
                    pass
    if not texts:
        defaults = [
            "A red heart shape with smooth curved edges, centered.",
            "Circular avatar: person with short black hair, dot eyes, blue shirt, minimal style.",
            "Sunset beach with orange gradient sky, yellow sun on horizon, blue waves, tan sand.",
            "Cute orange fox logo with triangular head, white chest, facing right, flat style.",
            "Simple bird silhouette: oval body, triangular beak, facing right.",
            "Simple house: red roof, beige body, blue windows, green ground at bottom.",
        ]
        texts = [[t] for t in defaults]
    return texts


def get_example_images() -> List[str]:
    ex_dir = find_examples_dir()
    if not ex_dir:
        return []
    imgs: List[str] = []
    for name in os.listdir(ex_dir):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            imgs.append(os.path.join(ex_dir, name))
    imgs.sort()
    return imgs


def create_gallery_html(candidates: Optional[List[Dict[str, Any]]]) -> str:
    if not candidates:
        return '<div style="text-align:center;color:#94a3b8;padding:50px;background:#0f172a;border-radius:14px;border:1px solid rgba(255,255,255,0.08);">No candidates generated yet.</div>'
    items = []
    for cand in candidates:
        svg_str = cand.get("svg", "")
        if svg_str and "viewBox" not in svg_str:
            svg_str = svg_str.replace("<svg", f'<svg viewBox="0 0 {TARGET_IMAGE_SIZE} {TARGET_IMAGE_SIZE}"', 1)
        items.append(
            f"""
        <div style="
            background: #0b1220;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 12px;
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        " onmouseover="this.style.transform='scale(1.02)';this.style.boxShadow='0 10px 30px rgba(3,7,18,0.6)';"
           onmouseout="this.style.transform='scale(1)';this.style.boxShadow='none';">
            <div style="width: 180px; height: 180px; margin: 0 auto; display: flex; align-items: center; justify-content: center; overflow: hidden; background:#05060b; border-radius: 10px;">
                {svg_str}
            </div>
            <div style="margin-top: 10px; font-size: 12px; color: #cbd5f5;">
                #{cand.get("index", '?')} | {cand.get("path_count", '?')} paths
            </div>
        </div>
        """
        )
    return f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        padding: 18px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.06);
    ">
        {''.join(items)}
    </div>
    """


def format_svg_code(candidates: Optional[List[Dict[str, Any]]]) -> str:
    if not candidates:
        return "<!-- No candidates -->"
    blocks = []
    for cand in candidates:
        blocks.append(
            f"<!-- ===== Candidate {cand.get('index', '?')} | Paths: {cand.get('path_count', '?')} ===== -->\n{cand.get('svg','')}"
        )
    return "\n\n".join(blocks)


def format_status(result: Dict[str, Any]) -> str:
    status = result.get("status") or "unknown"
    elapsed = result.get("elapsed_ms")
    model_size = result.get("model_size") or DEFAULT_MODEL_SIZE
    num = result.get("num_candidates", 0)
    message = result.get("message")
    base = f"{'✅' if status == 'ok' else '⚠️'} {status} | Model: {model_size} | Candidates: {num}"
    if elapsed:
        base += f" | {elapsed} ms"
    if message and status != "ok":
        base += f" | {message}"
    return base


def error_panel(message: str) -> str:
    return f'<div style="text-align:center;color:#fecaca;padding:40px;background:rgba(127,29,29,0.35);border:1px solid rgba(248,113,113,0.4);border-radius:14px;">{message}</div>'


def handle_text_submit(
    text: str,
    model_size: str,
    num_candidates: int,
    max_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
):
    env_err = ensure_env_ready()
    if env_err:
        return error_panel(env_err), "<!-- env error -->", env_err
    if not text.strip():
        msg = "Please provide a prompt."
        return error_panel(msg), "<!-- empty prompt -->", msg
    try:
        result = runsync(
            task_type="text-to-svg",
            text=text,
            model_size=model_size,
            num_candidates=int(num_candidates),
            max_length=int(max_length),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            return_png=True,
        )
        gallery = create_gallery_html(result.get("candidates"))
        svg_code = format_svg_code(result.get("candidates"))
        status_msg = format_status(result)
        if not result.get("candidates"):
            gallery = error_panel("No valid SVG generated. Try adjusting parameters or rephrasing.")
        return gallery, svg_code, status_msg
    except RunpodClientError as e:
        msg = f"ERROR: {e.message}"
        return error_panel(msg), "<!-- error -->", msg
    except Exception as e:
        msg = f"ERROR: {str(e)}"
        return error_panel(msg), "<!-- error -->", msg


def handle_image_submit(
    image: Optional[Image.Image],
    model_size: str,
    num_candidates: int,
    max_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    replace_background: bool,
):
    env_err = ensure_env_ready()
    if env_err:
        return error_panel(env_err), "<!-- env error -->", None, env_err
    if image is None:
        msg = "Please upload an image."
        return error_panel(msg), "<!-- missing image -->", None, msg
    try:
        img_b64 = encode_image_to_base64(image)
        result = runsync(
            task_type="image-to-svg",
            image_base64=img_b64,
            model_size=model_size,
            num_candidates=int(num_candidates),
            max_length=int(max_length),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            replace_background=replace_background,
            return_png=True,
        )
        gallery = create_gallery_html(result.get("candidates"))
        svg_code = format_svg_code(result.get("candidates"))
        status_msg = format_status(result)
        processed_img = decode_png_base64(result.get("processed_input_png_base64")) or image
        if not result.get("candidates"):
            gallery = error_panel("No valid SVG generated. Try toggling background replacement or new parameters.")
        return gallery, svg_code, processed_img, status_msg
    except RunpodClientError as e:
        msg = f"ERROR: {e.message}"
        return error_panel(msg), "<!-- error -->", None, msg
    except Exception as e:
        msg = f"ERROR: {str(e)}"
        return error_panel(msg), "<!-- error -->", None, msg


def build_ui():
    env_err = ensure_env_ready()
    example_texts = get_example_texts()
    example_images = get_example_images()

    with gr.Blocks(title="OmniSVG Runpod Client", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div class="header-container">
                <h1>OmniSVG Runpod Client</h1>
                <p>Stay in sync with the OmniSVG demo UI while driving the Runpod endpoint (4B / 8B ready).</p>
            </div>
            """
        )
        if env_err:
            gr.HTML(
                f"""
                <div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;padding:12px 15px;margin:15px 0;">
                    ⚠️ {env_err}
                </div>
                """
            )
        gr.HTML(
            """
            <div class="gpu-notice">
                🎲 Each run is unique—click generate again or tweak parameters to explore more candidates.
            </div>
            """
        )

        with gr.Tabs():
            with gr.TabItem("Text-to-SVG"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=320):
                        text_input = gr.Textbox(
                            label="Description",
                            placeholder="Describe your SVG with shapes and colors...",
                            lines=4,
                        )
                        with gr.Group(elem_classes=["settings-group"]):
                            text_model = gr.Dropdown(
                                choices=AVAILABLE_MODEL_SIZES,
                                value=DEFAULT_MODEL_SIZE,
                                label="Model Size",
                                info="8B for best quality, 4B for speed",
                            )
                            text_num_candidates = gr.Slider(
                                minimum=1,
                                maximum=MAX_NUM_CANDIDATES,
                                value=DEFAULT_NUM_CANDIDATES,
                                step=1,
                                label="Number of Candidates",
                            )
                            text_max_length = gr.Slider(
                                minimum=MAX_LENGTH_MIN,
                                maximum=MAX_LENGTH_MAX,
                                value=MAX_LENGTH_DEFAULT,
                                step=64,
                                label="Max Length",
                            )
                            with gr.Accordion("Advanced Parameters", open=False):
                                text_temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=TASK_CONFIGS["text-to-svg-icon"].get("default_temperature", 0.5),
                                    step=0.05,
                                    label="Temperature",
                                )
                                text_top_p = gr.Slider(
                                    minimum=0.5,
                                    maximum=1.0,
                                    value=TASK_CONFIGS["text-to-svg-icon"].get("default_top_p", 0.9),
                                    step=0.02,
                                    label="Top-P",
                                )
                                text_top_k = gr.Slider(
                                    minimum=10,
                                    maximum=100,
                                    value=TASK_CONFIGS["text-to-svg-icon"].get("default_top_k", 60),
                                    step=5,
                                    label="Top-K",
                                )
                                text_rep_penalty = gr.Slider(
                                    minimum=1.0,
                                    maximum=1.5,
                                    value=TASK_CONFIGS["text-to-svg-icon"].get("default_repetition_penalty", 1.03),
                                    step=0.01,
                                    label="Repetition Penalty",
                                )
                        text_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                        text_button = gr.Button("Generate SVG", variant="primary", elem_classes=["primary-btn"])
                        if example_texts:
                            gr.Examples(examples=example_texts, inputs=[text_input], label="Example Prompts")

                    with gr.Column(scale=2, min_width=500):
                        text_gallery = gr.HTML(
                            value='<div style="text-align:center;color:#94a3b8;padding:50px;background:#0f172a;border-radius:14px;border:1px solid rgba(255,255,255,0.08);">Generated SVGs will appear here.</div>'
                        )
                        text_svg_code = gr.Code(label="SVG Code", language="html", lines=15, elem_classes=["code-output"])

                text_button.click(
                    fn=handle_text_submit,
                    inputs=[
                        text_input,
                        text_model,
                        text_num_candidates,
                        text_max_length,
                        text_temperature,
                        text_top_p,
                        text_top_k,
                        text_rep_penalty,
                    ],
                    outputs=[text_gallery, text_svg_code, text_status],
                    queue=True,
                )

            with gr.TabItem("Image-to-SVG"):
                gr.HTML(IMAGE_TIPS_HTML)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=320):
                        image_input = gr.Image(
                            label="Upload Image",
                            type="pil",
                            image_mode="RGBA",
                            height=220,
                            sources=["upload", "clipboard"],
                            elem_classes=["input-image"],
                        )
                        with gr.Group(elem_classes=["settings-group"]):
                            img_model = gr.Dropdown(
                                choices=AVAILABLE_MODEL_SIZES,
                                value=DEFAULT_MODEL_SIZE,
                                label="Model Size",
                            )
                            img_num_candidates = gr.Slider(
                                minimum=1,
                                maximum=MAX_NUM_CANDIDATES,
                                value=DEFAULT_NUM_CANDIDATES,
                                step=1,
                                label="Number of Candidates",
                            )
                            img_replace_bg = gr.Checkbox(label="Replace non-white background", value=True)
                            img_max_length = gr.Slider(
                                minimum=MAX_LENGTH_MIN,
                                maximum=MAX_LENGTH_MAX,
                                value=MAX_LENGTH_DEFAULT,
                                step=64,
                                label="Max Length",
                            )
                            with gr.Accordion("Advanced Parameters", open=False):
                                img_temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=TASK_CONFIGS["image-to-svg"].get("default_temperature", 0.3),
                                    step=0.05,
                                    label="Temperature",
                                )
                                img_top_p = gr.Slider(
                                    minimum=0.5,
                                    maximum=1.0,
                                    value=TASK_CONFIGS["image-to-svg"].get("default_top_p", 0.9),
                                    step=0.02,
                                    label="Top-P",
                                )
                                img_top_k = gr.Slider(
                                    minimum=10,
                                    maximum=100,
                                    value=TASK_CONFIGS["image-to-svg"].get("default_top_k", 50),
                                    step=5,
                                    label="Top-K",
                                )
                                img_rep_penalty = gr.Slider(
                                    minimum=1.0,
                                    maximum=1.5,
                                    value=TASK_CONFIGS["image-to-svg"].get("default_repetition_penalty", 1.05),
                                    step=0.01,
                                    label="Repetition Penalty",
                                )
                        image_status = gr.Textbox(label="Status", value="Ready", interactive=False)
                        image_button = gr.Button("Generate SVG", variant="primary", elem_classes=["primary-btn"])
                        if example_images:
                            gr.Examples(examples=example_images, inputs=[image_input], label="Example Images")

                    with gr.Column(scale=2, min_width=500):
                        image_processed = gr.Image(label="Processed Input Preview", type="pil", height=180)
                        image_gallery = gr.HTML(
                            value='<div style="text-align:center;color:#94a3b8;padding:50px;background:#0f172a;border-radius:14px;border:1px solid rgba(255,255,255,0.08);">Generated SVGs will appear here.</div>'
                        )
                        image_svg_code = gr.Code(label="SVG Code", language="html", lines=12, elem_classes=["code-output"])

                image_button.click(
                    fn=handle_image_submit,
                    inputs=[
                        image_input,
                        img_model,
                        img_num_candidates,
                        img_max_length,
                        img_temperature,
                        img_top_p,
                        img_top_k,
                        img_rep_penalty,
                        img_replace_bg,
                    ],
                    outputs=[image_gallery, image_svg_code, image_processed, image_status],
                    queue=True,
                )

        gr.HTML(TIPS_HTML_BOTTOM)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniSVG Runpod Client UI")
    parser.add_argument("--listen", default="0.0.0.0", help="Listen address")
    parser.add_argument("--port", type=int, default=7860, help="Port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share")
    parser.add_argument("--debug", action="store_true", help="Show Gradio errors")
    args = parser.parse_args()
    app = build_ui()
    app.launch(server_name=args.listen, server_port=args.port, share=args.share, show_error=args.debug)

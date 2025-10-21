import os
import time
import io
from typing import Any, Dict, Optional

import runpod
from PIL import Image
import cairosvg

import service

# Avoid tokenizer parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODELS_READY = False


def ensure_models_loaded() -> None:
    """
    Ensure heavy models are loaded once per cold start unless dummy mode is enabled.
    """
    global MODELS_READY
    if MODELS_READY:
        return

    # In Hub tests we may not have weights/models; dummy mode skips loading.
    if os.environ.get("ENABLE_DUMMY", "true").lower() == "true":
        MODELS_READY = True
        return

    # Reuse service's global lock to avoid racing GPU memory on warm requests
    with service.gen_lock:
        if MODELS_READY:
            return
        service.load_models_once()
        MODELS_READY = True


def build_dummy_svg(text: Optional[str], mode: str) -> str:
    """
    Produce a simple but valid SVG to pass hub tests without model init.
    """
    safe_text = (text or "").strip().replace("\n", " ")[:80]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#6c63ff"/>
      <stop offset="100%" stop-color="#00c2ff"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="200" height="200" fill="url(#g)"/>
  <rect x="8" y="8" width="184" height="184" rx="12" ry="12" fill="#ffffff" opacity="0.9"/>
  <text x="16" y="40" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" fill="#333">OmniSVG dummy</text>
  <text x="16" y="60" font-family="DejaVu Sans, Arial, sans-serif" font-size="10" fill="#555">mode: {mode}</text>
  <text x="16" y="80" font-family="DejaVu Sans, Arial, sans-serif" font-size="10" fill="#777">{safe_text}</text>
</svg>'''.replace("<", "<").replace(">", ">")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runpod Serverless handler.
    Expects event.input with keys:
      - task_type: "text-to-svg" | "image-to-svg"
      - text: optional (required for text-to-svg)
      - image_base64: optional (required for image-to-svg)
      - return_png: bool (default True)
    """
    t0 = time.time()
    try:
        payload = event.get("input") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            return {"error": "Missing or invalid 'input' payload."}

        task_type = str(payload.get("task_type", "")).strip()
        text = payload.get("text")
        image_base64 = payload.get("image_base64")
        return_png = payload.get("return_png", True)

        dummy = os.environ.get("ENABLE_DUMMY", "true").lower() == "true"

        # Dummy path for Hub tests / dry runs without heavy model init.
        if dummy:
            if task_type not in ("image-to-svg", "text-to-svg"):
                task_type = "text-to-svg" if text else "image-to-svg"
            svg_code = build_dummy_svg(text if task_type == "text-to-svg" else "image", task_type)
            resp = {
                "svg": svg_code,
                "elapsed_ms": int((time.time() - t0) * 1000),
                "dummy": True
            }
            if return_png:
                png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
                png_image = Image.open(io.BytesIO(png_data))
                resp["png_base64"] = service.pil_to_b64(png_image, format="PNG")
            return resp

        # Real path: ensure models are loaded once, then run inference.
        ensure_models_loaded()

        if task_type not in ("image-to-svg", "text-to-svg"):
            return {"error": "task_type must be 'image-to-svg' or 'text-to-svg'"}

        with service.gen_lock:
            if task_type == "text-to-svg":
                if not text:
                    return {"error": "text is required for text-to-svg"}
                input_ids, attention_mask, pixel_values, image_grid_thw = service.process_text_to_svg_inputs(text)
                svg_code, png_image = service.generate_svg(
                    input_ids, attention_mask, pixel_values, image_grid_thw, "text-to-svg"
                )
            else:
                if not image_base64:
                    return {"error": "image_base64 is required for image-to-svg"}
                image_pil = service.b64_to_pil(image_base64)
                processed = service.process_and_resize_image(image_pil)
                input_ids, attention_mask, pixel_values, image_grid_thw = service.process_image_to_svg_inputs(processed)
                svg_code, png_image = service.generate_svg(
                    input_ids, attention_mask, pixel_values, image_grid_thw, "image-to-svg"
                )

        if not svg_code or str(svg_code).startswith("Error"):
            return {"error": str(svg_code)}

        resp = {
            "svg": svg_code,
            "elapsed_ms": int((time.time() - t0) * 1000)
        }
        if png_image is not None and (return_png is None or return_png):
            resp["png_base64"] = service.pil_to_b64(png_image, format="PNG")
        return resp

    except Exception as e:
        return {"error": str(e)}


# Start Runpod serverless runtime
runpod.serverless.start({"handler": handler})
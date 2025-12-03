import io
import os
import time
from typing import Any, Dict, Optional

import cairosvg
import runpod
from PIL import Image

import service

# Avoid tokenizer parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_dummy_svg(text: Optional[str], mode: str) -> str:
    """Produce a simple but valid SVG to pass hub tests without heavy model init."""
    safe_text = (text or "").strip().replace("\n", " ")[:80]
    return f"""<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200' viewBox='0 0 200 200'>
  <defs>
    <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0%' stop-color='#6c63ff'/>
      <stop offset='100%' stop-color='#00c2ff'/>
    </linearGradient>
  </defs>
  <rect x='0' y='0' width='200' height='200' fill='url(#g)'/>
  <rect x='8' y='8' width='184' height='184' rx='12' ry='12' fill='#ffffff' opacity='0.9'/>
  <text x='16' y='40' font-family='DejaVu Sans, Arial, sans-serif' font-size='12' fill='#333'>OmniSVG dummy</text>
  <text x='16' y='60' font-family='DejaVu Sans, Arial, sans-serif' font-size='10' fill='#555'>mode: {mode}</text>
  <text x='16' y='80' font-family='DejaVu Sans, Arial, sans-serif' font-size='10' fill='#777'>{safe_text}</text>
</svg>"""


def build_dummy_response(task_type: str, payload: Dict[str, Any], elapsed_ms: int) -> Dict[str, Any]:
    svg_code = build_dummy_svg(payload.get("text") if task_type == "text-to-svg" else "image", task_type)
    return_png = payload.get("return_png", True)
    candidate = {"index": 1, "svg": svg_code, "path_count": 1}
    if return_png:
        png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
        png_image = Image.open(io.BytesIO(png_data))
        candidate["png_base64"] = service.pil_to_b64(png_image)
    result = {
        "status": "ok",
        "dummy": True,
        "task_type": task_type,
        "model_size": payload.get("model_size") or service.DEFAULT_MODEL_SIZE,
        "elapsed_ms": elapsed_ms,
        "parameters": {},
        "num_candidates": 1,
        "return_png": bool(return_png),
        "candidates": [candidate],
        "primary_svg": svg_code,
    }
    if return_png:
        result["primary_png_base64"] = candidate.get("png_base64")
    return result


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runpod Serverless handler.
    Accepts the same payload as service.PredictRequest, wrapped under event['input'].
    """
    t0 = time.time()
    try:
        payload = event.get("input") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            return {"error": "Missing or invalid 'input' payload."}

        task_type = str(payload.get("task_type", "")).strip()
        if task_type not in {"text-to-svg", "image-to-svg"}:
            return {"error": "task_type must be 'text-to-svg' or 'image-to-svg'"}

        if os.environ.get("ENABLE_DUMMY", "true").lower() == "true":
            return build_dummy_response(task_type, payload, int((time.time() - t0) * 1000))

        request_model = service.PredictRequest(**payload)
        result = service.run_generation(request_model)
        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

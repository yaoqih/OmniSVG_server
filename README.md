# OmniSVG_server

[![Runpod](https://api.runpod.io/badge/yaoqih/OmniSVG_server)](https://console.runpod.io/hub/yaoqih/OmniSVG_server)

Serverless SVG generation from text or images, powered by Qwen-VL and a custom sketch decoder. This repo includes a Runpod Serverless handler and Hub configuration for one-click deployment.

## Components

- Serverless handler: see [`runpod_handler.handler()`](runpod_handler.py:36)
- Core service and inference utilities:
  - [`service.load_models_once()`](service.py:192) — lazy, one-time model init
  - [`service.process_text_to_svg_inputs()`](service.py:68) — text-to-svg preprocessing
  - [`service.process_image_to_svg_inputs()`](service.py:93) — image-to-svg preprocessing
  - [`service.generate_svg()`](service.py:126) — decoding to SVG and PNG
  - [`service.b64_to_pil()`](service.py:46), [`service.pil_to_b64()`](service.py:51) — I/O helpers
- Minimal Qwen vision utility: [`qwen_vl_utils.process_vision_info()`](qwen_vl_utils.py:4)
- Runpod Hub configuration: [.runpod/hub.json](.runpod/hub.json)
- Runpod Tests configuration: [.runpod/tests.json](.runpod/tests.json)
- Container build: [Dockerfile](Dockerfile)

## Runpod Serverless

Handler entry: [`runpod_handler.py`](runpod_handler.py)

Event input schema (JSON in `event.input`):
```json
{
  "task_type": "text-to-svg | image-to-svg",
  "text": "optional (required for text-to-svg)",
  "image_base64": "optional (required for image-to-svg)",
  "return_png": true
}
```

Response:
```json
{
  "svg": "<svg ...>...</svg>",
  "png_base64": "optional PNG preview (base64)",
  "elapsed_ms": 1234
}
```

Dummy mode:
- For Hub validations (no private weights), the handler supports `ENABLE_DUMMY=true` to return a valid, simple SVG without loading heavy models.
- For production, set `ENABLE_DUMMY=false` to run real inference with your weights and model.

## Environment Variables

These are read by [`service.load_models_once()`](service.py:192) and the handler:

- `WEIGHT_PATH` — path to your OmniSVG weights directory (default `/runpod-volume/OmniSVG`)
- `CONFIG_PATH` — path to the project config YAML (default `/workspace/config.yaml`)
- `QWEN_LOCAL_DIR` — local path (or HF repo id) for Qwen2.5-VL-3B-Instruct (default `/runpod-volume/Qwen2.5-VL-3B-Instruct`)
- `SVG_TOKENIZER_CONFIG` — path to tokenizer config YAML (default `/workspace/config.yaml`)
- `ENABLE_DUMMY` — whether to use dummy mode (default `true` in Hub; set to `false` for production)

You can override these in the Runpod Hub UI. Defaults are encoded in [.runpod/hub.json](.runpod/hub.json).

## Tests (Runpod Hub)

The Hub will use [.runpod/tests.json](.runpod/tests.json) to validate the listing. The default test:
- Sends a `text-to-svg` request with `"Hello world"`
- Sets `ENABLE_DUMMY=true` so the test succeeds without your private model/weights
- Targets GPU `"A40"` with CUDA 12.x

## Container

The [Dockerfile](Dockerfile) includes:
- Python 3.10 slim base
- System deps for CairoSVG (`libcairo2`, `libpango`, fonts)
- PyTorch CUDA 12.1 wheels + torchvision
- Python deps: `runpod`, `fastapi`, `uvicorn`, `transformers`, `Pillow`, `PyYAML`, `cairosvg`

Default CMD launches the serverless handler:
```
CMD ["python", "-u", "runpod_handler.py"]
```

## Local (optional)

You can still run the FastAPI service locally for debugging:
- Entry: [`service.py`](service.py)
- Run: `python service.py` (serves `/ping` and `/predict`)
- Note: Local environment must have matching CUDA/PyTorch and system libs for CairoSVG.

## Deploy Steps

1) Ensure your model and weights are accessible in Runpod via a mounted volume or public HF repo.
2) Configure env vars in Hub (or use defaults in [.runpod/hub.json](.runpod/hub.json)).
3) Create a GitHub release to trigger Hub ingestion.
4) In production, disable dummy mode: `ENABLE_DUMMY=false`.

## License

No license specified in this repository.
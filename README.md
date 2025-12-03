# OmniSVG_server

[![Runpod](https://api.runpod.io/badge/yaoqih/OmniSVG_server)](https://console.runpod.io/hub/yaoqih/OmniSVG_server)

Serverless SVG generation from text or images, powered by Qwen-VL and a custom sketch decoder. This repo includes a Runpod Serverless handler and Hub configuration for one-click deployment.

## Components

- Serverless handler: [`handler.handler()`](handler.py:34) — wraps `service.run_generation` for Runpod
- Core service + inference helpers (`service.py`):
  - `ensure_model_loaded()` / `load_models()` — lazy multi-model (4B / 8B) init
  - `run_generation()` — shared entry for `/predict` and handler, returns multi-candidate payloads
  - `prepare_inputs`, `generate_candidates`, `b64_to_pil`/`pil_to_b64` utilities
- Minimal Qwen vision utility: [`qwen_vl_utils.process_vision_info()`](qwen_vl_utils.py:4)
- Client tooling: [`gradio_runpod.py`](gradio_runpod.py) + [`runpod_client.py`](runpod_client.py)
- Runpod Hub configuration: [.runpod/hub.json](.runpod/hub.json)
- Runpod Tests configuration: [.runpod/tests.json](.runpod/tests.json)
- Container build: [Dockerfile](Dockerfile)

## Runpod Serverless

Handler entry: [`handler.py`](handler.py)

Event input schema (JSON in `event.input`):
```json
{
  "task_type": "text-to-svg | image-to-svg",
  "text": "required for text-to-svg",
  "image_base64": "required for image-to-svg (base64 PNG/JPEG/WEBP, no data: prefix)",
  "model_size": "optional, defaults to config.default_model_size (4B | 8B)",
  "task_subtype": "optional, overrides icon/illustration auto-detect",
  "num_candidates": "optional, 1 ~ generation.max_num_candidates",
  "max_length": "optional, 256 ~ 2048",
  "temperature": "optional, float",
  "top_p": "optional, float",
  "top_k": "optional, int",
  "repetition_penalty": "optional, float",
  "replace_background": "optional, bool (image-to-svg)",
  "return_png": true
}
```

Response:
```json
{
  "status": "ok | no_valid_candidates | error",
  "task_type": "text-to-svg",
  "model_size": "4B",
  "subtype": "icon",
  "parameters": {
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_length": 512,
    "num_candidates": 1
  },
  "candidates": [
    {
      "index": 1,
      "path_count": 42,
      "svg": "<svg ...>...</svg>",
      "png_base64": "optional PNG preview"
    }
  ],
  "primary_svg": "<svg ...>",
  "primary_png_base64": "optional PNG preview",
  "processed_input_png_base64": "image preview (image-to-svg)",
  "elapsed_ms": 1234
}
```

Dummy mode:
- For Hub validations (no private weights), the handler supports `ENABLE_DUMMY=true` to return a valid, simple SVG without loading heavy models.
- For production, set `ENABLE_DUMMY=false` to run real inference with your weights and model.

## Environment Variables

Handled by `service.ensure_model_loaded()` / `handler.py`:

- `CONFIG_PATH` — config YAML path (default `/workspace/config.yaml`)
- `WEIGHT_PATH` — fallback OmniSVG weights path or HF repo (default `/runpod-volume/OmniSVG`)
- `WEIGHT_PATH_4B`, `WEIGHT_PATH_8B` — optional overrides for each model (default `/runpod-volume/OmniSVG1.1_4B` etc.)
- `QWEN_LOCAL_DIR` — fallback Qwen model path / repo id (default `/runpod-volume/Qwen2.5-VL-3B-Instruct`)
- `QWEN_MODEL_4B`, `QWEN_MODEL_8B` — optional overrides for each backbone (defaults `/runpod-volume/Qwen2.5-VL-3B-Instruct` and `/runpod-volume/Qwen2.5-VL-7B-Instruct`)
- `SVG_TOKENIZER_CONFIG` — tokenizer config path (default `/workspace/config.yaml`)
- `ENABLE_DUMMY` — return placeholder SVGs without loading weights (default `true` in Hub; set `false` for production)

Defaults are encoded in [.runpod/hub.json](.runpod/hub.json) and surfaced as editable env fields.

## Tests (Runpod Hub)

Hub validations use [.runpod/tests.json](.runpod/tests.json):
- Text + image smoke tests send the latest parameter set (model size, sampling knobs, background toggle).
- `ENABLE_DUMMY=true` is injected so tests succeed without your private weights.
- GPU target: `A40` + CUDA 12.x (adjust as needed).

## Container

The [Dockerfile](Dockerfile) includes:
- Python 3.10 slim base
- System deps for CairoSVG (`libcairo2`, `libpango`, fonts)
- PyTorch CUDA 12.1 wheels + torchvision
- Python deps: `runpod`, `fastapi`, `uvicorn`, `transformers`, `Pillow`, `PyYAML`, `cairosvg`

Default CMD launches the serverless handler:
```
CMD ["python", "-u", "handler.py"]
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
## Gradio 客户端（Runpod 后端）

该客户端提供一个轻量的 Gradio 前端，直接通过 HTTP（requests）调用 Runpod Serverless 队列端点进行推理。无需本地加载大模型，亦不会调用本仓库的本地服务（不会触发 `service.load_models_once` 等）。

- 代码入口：`gradio_runpod.py`
- Runpod HTTP 客户端：`runpod_client.py`，核心方法：
  - 同步调用：[`runpod_client.runsync()`](runpod_client.py:87)
  - 异步轮询：[`runpod_client.run_async()`](runpod_client.py:121)

### 环境变量

在启动前配置以下环境变量（不要在代码中硬编码密钥）：

- `RUNPOD_API_KEY` — 你的 Runpod API Key
- `ENDPOINT_ID` — Runpod 队列端点 ID

Linux / macOS:
```bash
export RUNPOD_API_KEY="rp_xxx_your_api_key"
export ENDPOINT_ID="xxxxxxxxxxxxxxxx"
```

Windows PowerShell:
```powershell
$env:RUNPOD_API_KEY="rp_xxx_your_api_key"
$env:ENDPOINT_ID="xxxxxxxxxxxxxxxx"
```

### 安装与启动

- 安装依赖：
  ```bash
  pip install -r requirements_client.txt
  # 若本地未安装 gradio，请执行：
  # pip install gradio
  ```
- 启动 Gradio 客户端：
  ```bash
  python gradio_runpod.py --listen 0.0.0.0 --port 7860
  # 可选参数：
  #   --share   启用 Gradio 对外分享链接
  #   --debug   显示详细错误（Gradio show_error）
  ```

启动后浏览器访问对应地址（例如 `http://127.0.0.1:7860`）。

### 界面与使用

客户端包含与服务端一致的两个 Tab，并暴露所有关键参数：

1) **Text-to-SVG**
   - 输入：prompt 文本（gr.Textbox）
   - 模型选择：4B/8B 下拉框
   - 采样设置：候选数量、max_length、temperature、top_p、top_k、repetition_penalty（折叠在高级设置）
   - 行为：调用 [`runpod_client.runsync()`](runpod_client.py:87)，也可以扩展为 `run_async`

2) **Image-to-SVG**
   - 输入：图像（gr.Image，type="pil"，image_mode="RGBA"）
   - 设置：同 Text 视图外加 `replace_background` 开关
   - 行为：将图像编码为 base64 后调用 Runpod 端点

输出组件包括：
- SVG 网格（HTML gallery）
- SVG 代码（gr.Code）
- PNG 预览（输入处理 + 主候选）
- 运行状态文本（model / elapsed / message）

### 请求与返回结构（客户端侧）

- 同步模式（runsync）：
  - POST `https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync?wait=120000`
  - Headers：
    - `accept: application/json`
    - `authorization: RUNPOD_API_KEY`（值为环境变量内容）
    - `content-type: application/json`
  - Body（示例）：
    ```json
    {
      "input": {
        "task_type": "text-to-svg",
        "text": "...",
        "model_size": "4B",
        "num_candidates": 1,
        "max_length": 512,
        "temperature": 0.4,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "return_png": true
      }
    }
    ```
  - 返回兼容两类格式：顶层业务字段或 `output` 包裹，客户端统一解析为  
    `{'status': str|null, 'svg': str|null, 'png_base64': str|null, 'candidates': list|null, 'parameters': dict|null, 'elapsed_ms': int|null, 'delayTime': int|null, 'executionTime': int|null, ...}`

- 异步模式（队列轮询）：
  - POST 提交：`https://api.runpod.ai/v2/{ENDPOINT_ID}/run`
  - 轮询 GET：`https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}`
  - 轮询策略：
    - 基础每 2s 轮询，可依据 `delayTime` 动态调整
    - 总超时默认 180s
    - 遇到 429：指数退避（基础 1s，倍增至最多 16s，附加 0-500ms 抖动）
    - 401/404/500：立即报错并停止

### 错误处理与提示

- 客户端将 HTTP 状态码与响应中的 `error` 字段合并为用户可读的错误信息。
- UI 顶部在环境变量缺失（`RUNPOD_API_KEY` 或 `ENDPOINT_ID`）时，会显示明显错误并阻止发起请求。
- Gradio UI 中仅显示简化错误信息；原始响应片段可用于日志排查。

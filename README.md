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
  pip install -r requirements.txt
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

客户端包含两个 Tab，风格与交互参考本仓库服务端风格：

1) Text-to-SVG
- 输入：文本（gr.Textbox）
- 选项：return_png（默认 true）与执行方式（Dropdown，默认“同步(runsync)”）
- 行为：调用 [`runpod_client.runsync()`](runpod_client.py:87) 或 [`runpod_client.run_async()`](runpod_client.py:121) 并传入 `text`

2) Image-to-SVG
- 输入：图像（gr.Image，type="pil"，image_mode="RGBA"）
- 选项：return_png（默认 true）与执行方式（Dropdown，默认“同步(runsync)”）
- 行为：将图像按需编码为 PNG base64（不带 data: 前缀）后调用 Runpod 端点

输出组件（两种 Tab 相同）：
- SVG 代码：gr.Textbox（带复制按钮）
- PNG 预览：gr.Image（若返回 png_base64 则解码显示，否则为空）
- 辅助状态信息：gr.Markdown（展示 status、delayTime、executionTime、elapsed_ms（若存在））

顶部文案会提示可能的费用与时延（例如 A40 成本、冷启动与热请求差异），仅为提示不做计算。

### 请求与返回结构（客户端侧）

- 同步模式（runsync）：
  - POST `https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync?wait=120000`
  - Headers：
    - `accept: application/json`
    - `authorization: RUNPOD_API_KEY`（值为环境变量内容）
    - `content-type: application/json`
  - Body：
    ```json
    {
      "input": {
        "task_type": "text-to-svg | image-to-svg",
        "text": "...",
        "image_base64": "...",
        "return_png": true
      }
    }
    ```
  - 返回兼容两类格式：顶层业务字段或 `output` 包裹，统一解析为：
    `{'svg': str, 'png_base64': str|null, 'elapsed_ms': int|null, 'status': str|null, 'delayTime': int|null, 'executionTime': int|null}`

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

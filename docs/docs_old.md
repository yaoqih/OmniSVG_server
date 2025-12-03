本仓库提供两种对外使用方式：
- Runpod Serverless 队列端点（推荐生产）：入口为 handler.handler()，由 runpod.serverless.start() 启动。
- 本地 FastAPI 服务（便于开发调试）：入口为 service.predict_endpoint()，请求/响应模型为 service.PredictRequest。
核心推理逻辑集中在 `service.run_generation()`，按需调用 `service.ensure_model_loaded()` 完成 4B/8B 模型的延迟加载。Serverless 冷启动在第一次真实请求时自动加载对应模型；Hub 验证或演示可开启 `ENABLE_DUMMY` 以跳过大模型加载。
注意：Serverless 队列端点遵循 Runpod 固定操作集（/runsync、/run、/status 等）；本地 FastAPI 则提供 /predict 与 /ping 路由。以下分别说明。
一、通用输入输出约定
- 内容类型：JSON。Serverless 必须把业务入参包裹在顶层 `input` 字段内。
- 字段说明：
  - `task_type`: 字符串，`"text-to-svg"` 或 `"image-to-svg"`（必填）。
  - `text`: 字符串；`task_type="text-to-svg"` 时必填。
  - `image_base64`: 字符串；`task_type="image-to-svg"` 时必填，需为图片的 base64（无需 `data:` 前缀）。
  - `model_size`: 字符串，可选，例 `4B`/`8B`；不填使用 `config.yaml` 的 `default_model_size`。
  - `task_subtype`: 字符串，可选，`"icon"` | `"illustration"`；留空时服务端会根据文本自动判别。
  - `num_candidates`: 整数，可选；生成候选数量（范围 1~`generation.max_num_candidates`）。
  - `max_length`: 整数，可选；最大生成长度（范围 256~2048）。
  - `temperature` / `top_p` / `top_k` / `repetition_penalty`: 采样参数，可选；不填则按任务类型加载默认值。
  - `replace_background`: 布尔，可选，仅 image-to-svg 使用；是否尝试将非白色背景替换成纯白。
  - `return_png`: 布尔，可选，默认 true；是否在响应中附带 PNG 预览（base64）。
- Serverless 请求体示例（顶层 input）：
```
{
  "input": {
    "task_type": "text-to-svg",
    "text": "a red circle with a black border",
    "model_size": "4B",
    "num_candidates": 2,
    "max_length": 768,
    "temperature": 0.4,
    "return_png": true
  }
}
```
- 典型响应（经 service.run_generation 标准化）：
```
{
  "status": "ok",
  "task_type": "text-to-svg",
  "model_size": "4B",
  "num_candidates": 2,
  "parameters": {
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_length": 768,
    "num_candidates": 2
  },
  "candidates": [
    {"index": 1, "path_count": 42, "svg": "<svg ...>", "png_base64": "..."},
    {"index": 2, "path_count": 38, "svg": "<svg ...>", "png_base64": "..."}
  ],
  "primary_svg": "<svg ...>",
  "primary_png_base64": "...",
  "elapsed_ms": 2345
}
```
二、Runpod Serverless 队列端点
- 基础路径：https://api.runpod.ai/v2/$ENDPOINT_ID
- 认证：HTTP 头 authorization: $RUNPOD_API_KEY
- 同步 vs 异步：
  - /runsync：同步，适用于短任务；结果保留 1 分钟（可通过 ?wait=ms 最长 5 分钟）。
  - /run：异步，立即返回 job id；完成后 30 分钟内可取结果。
1. /runsync（POST，同步执行）
- 请求大小上限：20 MB。
```bash
curl --request POST \
     --url https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
     -H "accept: application/json" \
     -H "authorization: $RUNPOD_API_KEY" \
     -H "content-type: application/json" \
     -d '{
           "input": {
             "task_type": "text-to-svg",
             "text": "Hello, world!",
             "model_size": "8B",
             "num_candidates": 2,
             "return_png": true
           }
         }'
```
Python（Runpod SDK）：
 import os
 import runpod
 
 runpod.api_key = os.getenv("RUNPOD_API_KEY")
 endpoint = runpod.Endpoint(os.getenv("ENDPOINT_ID"))
 
 result = endpoint.run_sync({
     "task_type": "text-to-svg",
     "text": "Hello, world!",
     "return_png": True
 }, timeout=60)
 print(result)
JavaScript（runpod-sdk）：
 import runpodSdk from "runpod-sdk";
 const { RUNPOD_API_KEY, ENDPOINT_ID } = process.env;
 const runpod = runpodSdk(RUNPOD_API_KEY);
 const endpoint = runpod.endpoint(ENDPOINT_ID);
 const result = await endpoint.runSync({
   input: {
     task_type: "text-to-svg",
     text: "Hello, world!",
     return_png: true
   },
   timeout: 60000
 });
 console.log(result);
2. /run（POST，异步执行）
- 请求大小上限：10 MB。
- 返回示例：{"id": "...", "status": "IN_QUEUE"}。
cURL：
 curl --request POST \
      --url https://api.runpod.ai/v2/$ENDPOINT_ID/run \
      -H "accept: application/json" \
      -H "authorization: $RUNPOD_API_KEY" \
      -H "content-type: application/json" \
      -d '{"input": {"task_type": "image-to-svg", "image_base64": "<BASE64>", "return_png": true}}'
Python（Runpod SDK）：
 import os, runpod
 runpod.api_key = os.getenv("RUNPOD_API_KEY")
 endpoint = runpod.Endpoint(os.getenv("ENDPOINT_ID"))
 req = endpoint.run({
   "task_type": "text-to-svg",
   "text": "Hello, async!"
 })
 print(req.id, req.status())
 output = req.output(timeout=60)
 print(output)
JavaScript（runpod-sdk）：
 import runpodSdk from "runpod-sdk";
 const runpod = runpodSdk(process.env.RUNPOD_API_KEY);
 const endpoint = runpod.endpoint(process.env.ENDPOINT_ID);
 const { id } = await endpoint.run({
   input: { task_type: "text-to-svg", text: "Hello" }
 });
 const status = await endpoint.status(id);
 console.log(status);
3. /status（GET，查询作业状态与结果）
cURL：
curl --request GET \
     --url https://api.runpod.ai/v2/$ENDPOINT_ID/status/YOUR_JOB_ID \
     -H "authorization: $RUNPOD_API_KEY"
响应包含 status、delayTime、executionTime，以及完成时的 `output`（即 `service.run_generation` 返回的完整 JSON）。
4. /cancel（POST，取消作业）
curl --request POST \
  --url https://api.runpod.ai/v2/$ENDPOINT_ID/cancel/YOUR_JOB_ID \
  -H "authorization: $RUNPOD_API_KEY"
5. /retry（POST，重试失败或超时作业）
curl --request POST \
     --url https://api.runpod.ai/v2/$ENDPOINT_ID/retry/YOUR_JOB_ID \
     -H "authorization: $RUNPOD_API_KEY"
6. /purge-queue（POST，清空等待队列）
curl --request POST \
     --url https://api.runpod.ai/v2/$ENDPOINT_ID/purge-queue \
     -H "authorization: $RUNPOD_API_KEY"
- 仅影响等待中的作业，进行中的作业不受影响。
7. /health（GET，端点健康状态）
curl --request GET \
     --url https://api.runpod.ai/v2/$ENDPOINT_ID/health \
     -H "authorization: $RUNPOD_API_KEY"
关于 /stream：本项目当前未实现流式增量输出；如需启用，请在模型推理侧实现流式 handler 并遵循 Runpod 流式协议。
三、本地 FastAPI 服务
- 启动：`python service.py`。
- 路由：
  - `GET /ping`：健康检查（返回当前设备、已加载模型等信息）。
  - `POST /predict`：推理；支持 `application/json` 与 `multipart/form-data`，字段与 Runpod 请求一致（图像可以通过 `image_base64` 或 `image` 文件上传）。
示例：
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "content-type: application/json" \
  -d '{
        "task_type": "text-to-svg",
        "text": "a red circle with a black border",
        "model_size": "4B",
        "num_candidates": 2,
        "return_png": true
      }'
```
四、参数验证与错误处理
- Serverless 与本地共享同一套校验：`task_type` / `text` / `image_base64` 缺失会立即报错，`model_size` 不在配置列表亦会报错。
- Service 返回结构里的 `status` 若非 `ok` 会附带 `message`；Serverless 层若出现异常仍以 `{"error": "..."} `形式返回。
- FastAPI 对无效请求给出 400，内部异常为 500。
- 常见平台级错误码：
  - 400 Bad Request：请求格式或参数错误
  - 401 Unauthorized：API Key 无效或无权限
  - 404 Not Found：ENDPOINT_ID 不存在
  - 429 Too Many Requests：触发限流，需退避重试
  - 500 Internal Server Error：端点内部异常，可查看日志
五、环境变量（可在 Runpod Hub/控制台配置）
- `CONFIG_PATH`：配置文件路径（默认 `./config.yaml`）。
- `WEIGHT_PATH`：通用权重目录/Hub Repo。
- `WEIGHT_PATH_4B`、`WEIGHT_PATH_8B`：可选，覆盖不同模型的权重路径。
- `QWEN_LOCAL_DIR`：通用 Qwen 模型目录/ID。
- `QWEN_MODEL_4B`、`QWEN_MODEL_8B`：可选，覆盖不同模型的 Qwen 源。
- `SVG_TOKENIZER_CONFIG`：SVG Tokenizer 配置，默认使用 `CONFIG_PATH`。
- `ENABLE_DUMMY`：是否启用假数据模式（默认 true，生产建议 false）。
六、速率限制与并发（平台侧）
- /runsync：基准 2000 次/10 秒；并发上限约 400
- /run：基准 1000 次/10 秒；并发约 200
- /status、/stream：基准 2000 次/10 秒；并发约 400
- /cancel：基准 100 次/10 秒；并发约 20
- /purge-queue：基准 2 次/10 秒
- 动态限流：有效上限=max(基准上限, 运行中 worker 数 × 每 worker 限额)。触发 429 时请指数退避重试。
七、高级选项（Serverless 顶层可选字段）
- webhook：作业完成时回调到你的 URL（平台会以 POST 发送与 /status 相同信息）。
- policy：单次作业的执行策略（executionTimeout、lowPriority、ttl）。
- s3Config：供 Worker 使用的 S3 兼容存储配置（本项目默认不使用，若需请在业务代码里读取）。
webhook 示例：
 {
   "input": { "task_type": "text-to-svg", "text": "..." },
   "webhook": "https://your-webhook-url.com"
 }
policy 示例：
 {
   "input": { "task_type": "text-to-svg", "text": "..." },
   "policy": {
     "executionTimeout": 900000,
     "lowPriority": false,
     "ttl": 3600000
   }
 }
s3Config 示例：
 {
   "input": { "task_type": "image-to-svg", "image_base64": "..." },
   "s3Config": {
     "accessId": "...",
     "accessSecret": "...",
     "bucketName": "...",
     "endpointUrl": "..."
   }
 }

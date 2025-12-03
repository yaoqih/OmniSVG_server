import os
import io
import json
import time
import random
import base64
from typing import Any, Dict, Optional, Tuple, Union
import requests



class RunpodClientError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, raw: Optional[Union[Dict[str, Any], str]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.raw = raw

API_BASE = "https://api.runpod.ai/v2"

def _get_env() -> Tuple[str, str]:
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("ENDPOINT_ID")
    if not api_key or not endpoint_id:
        raise RunpodClientError(
            "缺少环境变量：RUNPOD_API_KEY 或 ENDPOINT_ID",
            None,
            {"RUNPOD_API_KEY": bool(api_key), "ENDPOINT_ID": bool(endpoint_id)},
        )
    return api_key, endpoint_id

def _headers(api_key: str) -> Dict[str, str]:
    return {
        "accept": "application/json",
        "authorization": api_key,
        "content-type": "application/json",
    }

def encode_image_to_base64(image: Union["Image.Image", "np.ndarray"]) -> str:
    """将 PIL.Image 或 numpy 数组编码为 PNG base64（不含 data: 前缀）。"""
    try:
        from PIL import Image
        import numpy as np
    except Exception as e:
        raise RunpodClientError("图片编码依赖缺失：需要 Pillow 和 numpy", None, str(e))

    if image is None:
        raise RunpodClientError("未提供图像用于编码", None, None)

    if isinstance(image, Image.Image):
        img = image.convert("RGBA")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
        img = img.convert("RGBA")
    else:
        raise RunpodClientError(
            "不支持的图像类型：仅支持 PIL.Image 或 numpy.ndarray",
            None,
            {"type": str(type(image))},
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _parse_business_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    """统一解析输出为规范结构。兼容顶层和 output 包裹。"""
    data = obj or {}
    out = data.get("output") if isinstance(data.get("output"), dict) else data

    candidates = out.get("candidates")
    first_candidate = candidates[0] if isinstance(candidates, list) and candidates else {}

    svg = (
        out.get("primary_svg")
        or first_candidate.get("svg")
        or out.get("svg")
    )
    png_b64 = (
        out.get("primary_png_base64")
        or first_candidate.get("png_base64")
        or out.get("png_base64")
    )
    elapsed_ms = out.get("elapsed_ms")
    status = data.get("status") or out.get("status")
    delay_time = data.get("delayTime") if "delayTime" in data else out.get("delayTime")
    exec_time = data.get("executionTime") if "executionTime" in data else out.get("executionTime")

    return {
        "svg": svg if isinstance(svg, str) else (json.dumps(svg) if svg is not None else None),
        "png_base64": png_b64 if isinstance(png_b64, str) else None,
        "elapsed_ms": int(elapsed_ms) if isinstance(elapsed_ms, (int, float)) else None,
        "status": status if isinstance(status, str) else (str(status) if status is not None else None),
        "delayTime": int(delay_time) if isinstance(delay_time, (int, float)) else None,
        "executionTime": int(exec_time) if isinstance(exec_time, (int, float)) else None,
        "task_type": out.get("task_type"),
        "model_size": out.get("model_size"),
        "subtype": out.get("subtype"),
        "parameters": out.get("parameters"),
        "num_candidates": out.get("num_candidates"),
        "candidates": candidates if isinstance(candidates, list) else None,
        "processed_input_png_base64": out.get("processed_input_png_base64"),
    }


def _build_input_payload(
    task_type: str,
    text: Optional[str],
    image_base64: Optional[str],
    return_png: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task_type": task_type,
        "text": text,
        "image_base64": image_base64,
        "return_png": bool(return_png),
    }
    if extra:
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
    return {"input": payload}

def runsync(
    task_type: str,
    text: Optional[str] = None,
    image_base64: Optional[str] = None,
    *,
    model_size: Optional[str] = None,
    task_subtype: Optional[str] = None,
    num_candidates: Optional[int] = None,
    max_length: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    replace_background: Optional[bool] = None,
    return_png: bool = True,
    wait_ms: int = 120000,
) -> Dict[str, Any]:
    """同步调用 Runpod runsync 端点。"""
    api_key, endpoint_id = _get_env()
    url = f"{API_BASE}/{endpoint_id}/runsync?wait={wait_ms}"
    extra = {
        "model_size": model_size,
        "task_subtype": task_subtype,
        "num_candidates": num_candidates,
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "replace_background": replace_background,
    }
    payload = _build_input_payload(task_type, text, image_base64, return_png, extra)
    try:
        resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=(10, max(wait_ms / 1000, 30)))
        if resp.status_code != 200:
            snippet = resp.text[:500]
            raise RunpodClientError(f"HTTP {resp.status_code}: 同步请求失败", resp.status_code, snippet)
        data = resp.json() if "application/json" in resp.headers.get("Content-Type", "") else {"output": {"raw_text": resp.text}}
        if isinstance(data, dict) and "error" in data:
            raise RunpodClientError(f"服务错误: {data.get('error')}", resp.status_code, data)
        parsed = _parse_business_output(data)
        if not parsed.get("svg"):
            # 尝试兼容非标准返回
            raise RunpodClientError("返回缺少 svg 字段", resp.status_code, data)
        return parsed
    except requests.RequestException as e:
        raise RunpodClientError("网络错误：无法连接 Runpod 端点", None, str(e))

def run_async(
    task_type: str,
    text: Optional[str] = None,
    image_base64: Optional[str] = None,
    *,
    model_size: Optional[str] = None,
    task_subtype: Optional[str] = None,
    num_candidates: Optional[int] = None,
    max_length: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    replace_background: Optional[bool] = None,
    return_png: bool = True,
    timeout_s: int = 180,
    base_interval_s: float = 2.0,
) -> Dict[str, Any]:
    """异步队列：提交任务后轮询 status。"""
    api_key, endpoint_id = _get_env()
    submit_url = f"{API_BASE}/{endpoint_id}/run"
    extra = {
        "model_size": model_size,
        "task_subtype": task_subtype,
        "num_candidates": num_candidates,
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "replace_background": replace_background,
    }
    payload = _build_input_payload(task_type, text, image_base64, return_png, extra)
    t0 = time.time()
    try:
        s = requests.Session()
        resp = s.post(submit_url, headers=_headers(api_key), json=payload, timeout=(10, 30))
        if resp.status_code != 200:
            snippet = resp.text[:500]
            raise RunpodClientError(f"HTTP {resp.status_code}: 提交任务失败", resp.status_code, snippet)
        sub = resp.json() if "application/json" in resp.headers.get("Content-Type", "") else {}
        job_id = sub.get("id") or sub.get("jobId") or sub.get("job_id")
        if not job_id:
            raise RunpodClientError("提交响应缺少 job_id", resp.status_code, sub)
        status_url = f"{API_BASE}/{endpoint_id}/status/{job_id}"
        backoff = 1.0
        while True:
            if time.time() - t0 > timeout_s:
                raise RunpodClientError("轮询超时", None, {"job_id": job_id})
            try:
                r = s.get(status_url, headers=_headers(api_key), timeout=(5, 30))
            except requests.RequestException as e:
                # 连接错误，轻微退避重试
                time.sleep(min(base_interval_s, backoff) + random.uniform(0, 0.5))
                continue
            if r.status_code == 429:
                # 指数退避 + 抖动
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(backoff * 2, 16.0)
                continue
            if r.status_code in (401, 404, 500):
                snippet = r.text[:500]
                raise RunpodClientError(f"HTTP {r.status_code}: 轮询失败", r.status_code, snippet)
            if r.status_code != 200:
                # 其他非常规状态，短暂等待后继续
                time.sleep(base_interval_s + random.uniform(0, 0.5))
                continue
            data = r.json() if "application/json" in r.headers.get("Content-Type", "") else {"output": {"raw_text": r.text}}
            status = str(data.get("status", "")).upper()
            delay_time = data.get("delayTime")
            exec_time = data.get("executionTime")
            if data.get("error"):
                raise RunpodClientError(f"服务错误: {data.get('error')}", r.status_code, data)
            if status in ("COMPLETED", "SUCCESS", "FINISHED"):
                parsed = _parse_business_output(data)
                # 补充状态与时间信息
                parsed["status"] = status or parsed.get("status")
                if isinstance(delay_time, (int, float)):
                    parsed["delayTime"] = int(delay_time)
                if isinstance(exec_time, (int, float)):
                    parsed["executionTime"] = int(exec_time)
                if not parsed.get("svg"):
                    raise RunpodClientError("返回缺少 svg 字段", r.status_code, data)
                return parsed
            if status in ("FAILED", "ERROR"):
                raise RunpodClientError("任务失败", r.status_code, data)
            # 动态调整轮询间隔
            sleep_s = base_interval_s
            if isinstance(delay_time, (int, float)):
                # 假设 delayTime 为毫秒，设置在 [1, 5] 秒范围内
                sleep_s = max(1.0, min(5.0, float(delay_time) / 1000.0))
            time.sleep(sleep_s + random.uniform(0, 0.5))
    except requests.RequestException as e:
        raise RunpodClientError("网络错误：无法连接 Runpod 端点", None, str(e))

def ensure_env_ready() -> Optional[str]:
    """检查环境变量是否齐全，返回错误文案或 None。"""
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("ENDPOINT_ID")
    if not api_key or not endpoint_id:
        return "环境变量未配置：请设置 RUNPOD_API_KEY 和 ENDPOINT_ID 后重试。"
    return None

if __name__ == "__main__":
    # 简单自检：不发起真实请求，仅验证环境
    err = ensure_env_ready()
    if err:
        print(err)
    else:
        print("环境变量已就绪，可在 gradio_runpod.py 中使用。")

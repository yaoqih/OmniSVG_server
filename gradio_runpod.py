import os
import io
import base64
import argparse
from typing import Optional
from PIL import Image
import gradio as gr
from runpod_client import runsync, encode_image_to_base64, ensure_env_ready, RunpodClientError
from dotenv import load_dotenv
load_dotenv()
def decode_png_base64(b64: Optional[str]) -> Optional[Image.Image]:
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        return None

def find_examples_dir() -> Optional[str]:
    """查找示例目录（若存在）"""
    for d in ["examples", "example", "assets/examples", "assets/example"]:
        if os.path.isdir(d):
            return d
    return None

def get_example_texts() -> list:
    """返回用于 gr.Examples 的文本示例列表（每项为 [text]）。"""
    ex_dir = find_examples_dir()
    texts = []
    if ex_dir:
        for fname in ("texts.txt", "prompts.txt"):
            p = os.path.join(ex_dir, fname)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            s = line.strip()
                            if s:
                                texts.append([s])
                    break
                except Exception:
                    pass
    if not texts:
        texts = [
        "A yellow t-shirt with a heart design represents love and positivity.",
        "A bright yellow emoji with a surprised expression and rosy cheeks hovers above a shadow.",
        "A brown coffee cup on a white saucer is seen from a top-down perspective.",
        "A cartoon firefighter in a red and yellow uniform represents safety and protection.",
        "A cute bunny face with pink ears rosy cheeks and a playful red tongue conveys charm and cheerfulness.",
        "A bearded man with orange hair and a mustache represents a hipster style portrait.",
        "A colorful ice cream popsicle with a hint of chocolate at the bottom on a stick.",
        "A light blue shopping bag features a white flower with a red center and scattered dots.",
        "A yellow phone icon and orange arrow on a blue smartphone screen symbolize an incoming call.",
        "A sad wilted flower with pink petals slumps over an orange cloud with a blue striped background.",
        "A cartoon character with dark blue hair and a mustache wears a blue suit against a light blue circular background.",
        "A blue bookmark icon with a white plus sign in the center.",
        "A computer monitor displays a bar graph with yellow orange and green bars.",
        "A blue and gray database icon is overlaid with a yellow star in the bottom right corner.",
        "An orange thermometer with a circular base represents temperature measurement.",
        "A green delivery truck icon with a checkmark symbolizing a completed delivery.",
        "A yellow t-shirt with a heart design represents love and positivity.",
        "A blue and gray microphone icon symbolizes audio recording or voice input.",
        "Cloud icon with an upward arrow symbolizes uploading or cloud storage.",
        "A brown chocolate bar is depicted in four square segments with a shiny glossy finish.",
        "A colorful moving truck icon with a red and orange cargo container.",
        "A light blue T-shirt icon is outlined with a bold blue border.",
        "A person in a blue shirt and dark pants stands with one hand in a pocket gesturing outward.",
        ]
    return [[text ]for text in texts]

def get_example_images() -> list:
    """返回示例图片的文件路径列表。"""
    ex_dir = find_examples_dir()
    if not ex_dir:
        return []
    imgs = []
    try:
        for name in os.listdir(ex_dir):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                imgs.append(os.path.join(ex_dir, name))
    except Exception:
        return []
    imgs.sort()
    return imgs

def handle_text_submit(text: str):
    env_err = ensure_env_ready()
    if env_err:
        return "环境未就绪", None
    try:
        res = runsync(task_type="text-to-svg", text=text, image_base64=None, return_png=True)
        svg = res.get("svg") or ""
        png = decode_png_base64(res.get("png_base64"))
        return svg, png
    except RunpodClientError as e:
        return f"ERROR: {e.message}", None
    except Exception as e:
        return f"ERROR: {str(e)}", None

def handle_image_submit(image: Image.Image):
    env_err = ensure_env_ready()
    if env_err:
        return "环境未就绪", None
    try:
        img_b64 = encode_image_to_base64(image) if image is not None else None
        if not img_b64:
            return "ERROR: 未提供图像", None
        res = runsync(task_type="image-to-svg", text=None, image_base64=img_b64, return_png=True)
        svg = res.get("svg") or ""
        png = decode_png_base64(res.get("png_base64"))
        return svg, png
    except RunpodClientError as e:
        return f"ERROR: {e.message}", None
    except Exception as e:
        return f"ERROR: {str(e)}", None

def build_ui():
    env_err = ensure_env_ready()
    with gr.Blocks(title="OmniSVG Demo Page", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# OmniSVG Demo Page")
        gr.Markdown("Generate SVG code from images or text descriptions")

        if env_err:
            gr.Markdown(f"⚠️ 环境错误：{env_err}", elem_id="env-error")

        with gr.Tab("Text-to-SVG"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp_text = gr.Textbox(label="文本输入", placeholder="For example：A yellow t-shirt with a heart design represents love and positivity.", lines=4)
                    examples_text = get_example_texts()
                    if examples_text:
                        gr.Examples(
                            examples=examples_text,
                            inputs=[inp_text],
                            label="示例文本",
                        )
                    btn_text = gr.Button("Generate SVG")
                with gr.Column(scale=1):
                    out_svg_t = gr.Textbox(label="SVG 代码", show_copy_button=True)
                    out_png_t = gr.Image(label="PNG 预览", image_mode="RGBA")
            btn_text.click(
                fn=handle_text_submit,
                inputs=[inp_text],
                outputs=[out_svg_t, out_png_t],
            )

        with gr.Tab("Image-to-SVG"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp_img = gr.Image(type="pil", image_mode="RGBA", label="输入图像")
                    ex_imgs = get_example_images()
                    if ex_imgs:
                        gr.Examples(
                            examples=ex_imgs,
                            inputs=[inp_img],
                            label="示例图片",
                        )
                    btn_img = gr.Button("Generate SVG")
                with gr.Column(scale=1):
                    out_svg_i = gr.Textbox(label="SVG 代码", show_copy_button=True)
                    out_png_i = gr.Image(label="PNG 预览", image_mode="RGBA")
            btn_img.click(
                fn=handle_image_submit,
                inputs=[inp_img],
                outputs=[out_svg_i, out_png_i],
            )
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio 客户端（Runpod 队列后端）")
    parser.add_argument("--listen", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="端口")
    parser.add_argument("--share", action="store_true", help="Gradio share")
    parser.add_argument("--debug", action="store_true", help="调试模式（显示错误）")
    args = parser.parse_args()
    app = build_ui()
    app.launch(server_name=args.listen, server_port=args.port, share=args.share, show_error=args.debug)
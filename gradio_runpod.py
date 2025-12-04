import argparse
import base64
import io
import os
from typing import Any, Dict, List, Optional

import gradio as gr
import yaml
from dotenv import load_dotenv
from PIL import Image

# 假设 runpod_client 在同一目录下，保持引用不变
from runpod_client import (
    RunpodClientError,
    encode_image_to_base64,
    ensure_env_ready,
    run_async,
)

load_dotenv()

# --- 配置加载 (保持原有逻辑不变) ---
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"CONFIG_PATH not found: {CONFIG_PATH}")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

image_cfg = _config.get("image", {})
TARGET_IMAGE_SIZE = image_cfg.get("target_size", 448)

model_cfg = _config.get("model", {})
MAX_LENGTH = int(model_cfg.get("max_length", 1024))
MIN_MAX_LENGTH = 256
MAX_MAX_LENGTH = 2048

AVAILABLE_MODEL_SIZES = list(_config.get("models", {}).keys())
if not AVAILABLE_MODEL_SIZES:
    raise ValueError("No models defined in config.yaml")
DEFAULT_MODEL_SIZE = _config.get("default_model_size", AVAILABLE_MODEL_SIZES[0])

task_config = _config.get("task_configs", {})
TASK_CONFIGS = {
    "text-to-svg-icon": task_config.get("text_to_svg_icon", {
        "default_temperature": 0.5, "default_top_p": 0.88, "default_top_k": 50, "default_repetition_penalty": 1.05
    }),
    "text-to-svg-illustration": task_config.get("text_to_svg_illustration", {
        "default_temperature": 0.6, "default_top_p": 0.90, "default_top_k": 60, "default_repetition_penalty": 1.03
    }),
    "image-to-svg": task_config.get("image_to_svg", {
        "default_temperature": 0.3, "default_top_p": 0.90, "default_top_k": 50, "default_repetition_penalty": 1.05
    }),
}

gen_config = _config.get("generation", {})
DEFAULT_NUM_CANDIDATES = max(1, gen_config.get("default_num_candidates", 1))
MAX_NUM_CANDIDATES = max(DEFAULT_NUM_CANDIDATES, gen_config.get("max_num_candidates", 4))

try:
    RUNPOD_POLL_TIMEOUT_S = max(600, int(os.environ.get("RUNPOD_POLL_TIMEOUT_S", "480")))
except ValueError:
    RUNPOD_POLL_TIMEOUT_S = 480
try:
    RUNPOD_POLL_BASE_INTERVAL_S = max(1.0, float(os.environ.get("RUNPOD_POLL_BASE_INTERVAL_S", "2.0")))
except ValueError:
    RUNPOD_POLL_BASE_INTERVAL_S = 2.0


# --- 现代化 CSS ---
# 为浅色 & 深色模式提供统一的高对比度视觉样式
CUSTOM_CSS = """
:root {
    --surface-card: rgba(255, 255, 255, 0.92);
    --surface-alt: rgba(249, 250, 251, 0.9);
    --surface-grid: rgba(241, 245, 249, 0.8);
    --border-elevated: rgba(15, 23, 42, 0.08);
    --text-strong: #0f172a;
    --text-subtle: rgba(15, 23, 42, 0.66);
    --accent-gradient: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
}

.dark {
    --surface-card: rgba(15, 23, 42, 0.92);
    --surface-alt: rgba(22, 30, 54, 0.78);
    --surface-grid: rgba(30, 41, 59, 0.72);
    --border-elevated: rgba(148, 163, 184, 0.35);
    --text-strong: #f8fafc;
    --text-subtle: rgba(226, 232, 240, 0.78);
    --accent-gradient: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
}

body {
    color: var(--text-strong);
    background-color: var(--background-fill-primary);
    background-image: radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.08), transparent 45%), radial-gradient(circle at 90% 10%, rgba(14, 165, 233, 0.08), transparent 40%);
}

.dark body {
    color: var(--text-strong);
    background-color: var(--background-fill-primary);
    background-image: radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.12), transparent 45%), radial-gradient(circle at 90% 10%, rgba(99, 102, 241, 0.12), transparent 40%);
}

.gradio-container {
    color: var(--text-strong);
}

/* 头部样式 */
.header-container {
    text-align: center;
    margin-bottom: 28px;
    padding: 28px;
    background: linear-gradient(145deg, rgba(37, 99, 235, 0.12), rgba(14, 165, 233, 0.06));
    border-radius: 18px;
    border: 1px solid rgba(37, 99, 235, 0.18);
    color: var(--text-strong);
}
.header-logos {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 22px;
    flex-wrap: wrap;
    margin: 18px 0 10px 0;
}
.header-logos img {
    height: 44px;
    width: auto;
    display: block;
}
.header-logos svg {
    height: 34px;
    width: auto;
    display: block;
    filter: drop-shadow(0 6px 14px rgba(15, 23, 42, 0.12));
}
.dark .header-logos img,
.dark .header-logos svg {
    filter: drop-shadow(0 8px 18px rgba(0, 0, 0, 0.45));
}
.header-badge {
    display: inline-flex;
    padding: 4px 14px;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.08);
    font-size: 0.85em;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-strong);
}
.dark .header-badge {
    background: rgba(248, 250, 252, 0.1);
    color: #f8fafc;
}
.header-container h1 {
    margin: 14px 0 0 0;
    font-size: 2.2em;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.header-container p {
    margin: 10px auto 0 auto;
    opacity: 0.9;
    font-size: 1.02em;
    max-width: 540px;
    color: var(--text-subtle);
}

/* 参数面板样式 */
.settings-group {
    background: var(--surface-card) !important;
    border: 1px solid var(--border-elevated) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    margin-top: 16px !important;
    color: var(--text-strong);
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
    backdrop-filter: blur(16px);
}

/* SVG 展示卡片网格 */
.svg-gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    padding: 16px;
    background: var(--surface-alt);
    border-radius: 16px;
    border: 1px solid var(--border-elevated);
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
}

/* 单个 SVG 卡片 */
.svg-card {
    background: var(--surface-card);
    border: 1px solid var(--border-elevated);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    display: flex;
    flex-direction: column;
    align-items: center;
    color: var(--text-strong);
}
.svg-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 25px 50px rgba(15, 23, 42, 0.15);
    border-color: rgba(37, 99, 235, 0.35);
}
.svg-card svg {
    width: 100%;
    height: 100%;
}

/* 棋盘格背景：让透明 SVG 在深色/浅色下都可见 */
.svg-preview-box {
    width: 180px;
    height: 180px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    border-radius: 8px;
    background-color: #dedede;
    background-image:
        linear-gradient(45deg, #cdcdcd 25%, transparent 25%),
        linear-gradient(-45deg, #cdcdcd 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #cdcdcd 75%),
        linear-gradient(-45deg, transparent 75%, #cdcdcd 75%);
    background-size: 20px 20px;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
}
/* 深色模式下稍微暗一点的棋盘格，防止太刺眼 */
.dark .svg-preview-box {
    background-color: #2f2f2f;
    background-image:
        linear-gradient(45deg, #3d3d3d 25%, transparent 25%),
        linear-gradient(-45deg, #3d3d3d 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #3d3d3d 75%),
        linear-gradient(-45deg, transparent 75%, #3d3d3d 75%);
}

.svg-meta {
    margin-top: 12px;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-subtle);
}

/* 代码区域优化 */
.code-output textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 12px !important;
    border-radius: 10px !important;
    border: 1px solid var(--border-elevated) !important;
    background: var(--surface-card) !important;
    color: var(--text-strong) !important;
}

/* 输入图片区域 */
.input-image {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid var(--border-elevated);
}

.tips-box {
    margin-top: 32px;
    background: var(--surface-alt);
    border-radius: 18px;
    border: 1px solid var(--border-elevated);
    padding: 24px 28px;
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
    color: var(--text-strong);
}
.tips-box h3 {
    margin: 0 0 12px 0;
    font-weight: 600;
}
.tip-card {
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    border: 1px solid var(--border-elevated);
    color: var(--text-strong);
    background: var(--surface-card);
}
.tip-card ul {
    margin: 8px 0 0 0;
    padding-left: 22px;
}
.tip-card > strong {
    display: block;
    margin-bottom: 6px;
}
.tip-card ul {
    list-style: disc;
    list-style-position: inside;
}
.tip-card ul li {
    margin-bottom: 6px;
    color: var(--text-subtle);
}
.tip-card ul li strong {
    display: inline;
    color: var(--text-strong);
}

.red-box,
.blue-box,
.green-box {
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    border: 1px solid var(--border-elevated);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(148, 163, 184, 0.08));
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
}
.red-box {
    border-color: rgba(248, 113, 113, 0.25);
}
.blue-box {
    border-color: rgba(59, 130, 246, 0.25);
}
.green-box {
    border-color: rgba(34, 197, 94, 0.25);
}
.tip-category {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(148, 163, 184, 0.08));
    border-radius: 12px;
    padding: 16px;
    border: 1px solid var(--border-elevated);
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
}
.tip-category.icons {
}
.tip-category.animals {
}
.tip-category.objects {
}
.tip-category h4 {
    margin: 0 0 8px 0;
    color: var(--text-strong);
}
.tip-category p {
    margin: 0 0 10px 0;
    color: var(--text-subtle);
}
.tip-category code {
    background: var(--surface-alt);
    padding: 2px 6px;
    border-radius: 6px;
    margin-right: 4px;
    display: inline-block;
    font-size: 0.85em;
}
.example-prompt {
    background: var(--surface-alt);
    border-radius: 8px;
    padding: 8px 10px;
    margin: 6px 0;
    font-family: var(--font-mono);
    font-size: 0.9em;
    border: 1px solid var(--border-elevated);
}
.red-tip {
    color: #f97316;
    font-weight: 600;
}

.info-banner {
    background: var(--surface-card);
    border: 1px solid var(--border-elevated);
    border-radius: 12px;
    padding: 12px 15px;
    color: var(--text-strong);
    box-shadow: 0 16px 35px rgba(15, 23, 42, 0.08);
}
.info-banner.warning {
    border-left: 4px solid #f87171;
}
.info-banner.info {
    border-left: 4px solid #3b82f6;
}
.info-banner.success {
    border-left: 4px solid #22c55e;
}
.info-banner strong {
    color: inherit;
}

.prompt-structure {
    margin-top: 16px;
    padding: 14px;
    border-radius: 10px;
    border-left: 4px solid #4caf50;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(148, 163, 184, 0.08));
    border: 1px solid var(--border-elevated);
}
.prompt-example {
    background: var(--surface-alt);
    padding: 10px 14px;
    border-radius: 8px;
    font-family: var(--font-mono);
    font-size: 0.9em;
    border: 1px solid var(--border-elevated);
    margin-top: 10px;
}

.tip-card.tip-orange {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.18), rgba(251, 191, 36, 0.08));
    border-color: rgba(251, 191, 36, 0.4);
}
.dark .tip-card.tip-orange {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.28), rgba(251, 191, 36, 0.1));
}
.tip-card.tip-green {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.16), rgba(16, 185, 129, 0.06));
    border-color: rgba(16, 185, 129, 0.35);
}
.dark .tip-card.tip-green {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.28), rgba(16, 185, 129, 0.1));
}
.tip-card.tip-blue {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(59, 130, 246, 0.06));
    border-color: rgba(59, 130, 246, 0.35);
}
.dark .tip-card.tip-blue {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(59, 130, 246, 0.12));
}

.tip-highlight {
    margin-top: 16px;
    padding: 18px;
    border-radius: 14px;
    background: var(--surface-card);
    border: 1px dashed var(--border-elevated);
    color: var(--text-strong);
}
.tip-code {
    background: var(--surface-alt);
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 10px;
    font-family: var(--font-mono);
    border: 1px solid var(--border-elevated);
    font-size: 0.92em;
}
.tip-note {
    margin-top: 10px;
    color: var(--text-subtle);
    font-size: 0.95em;
}
.image-tip-card {
    margin-bottom: 18px;
}

.gpu-notice {
    background: var(--surface-card);
    border: 1px solid var(--border-elevated);
    border-radius: 16px;
    padding: 18px 22px;
    margin: 22px 0;
    display: flex;
    align-items: center;
    gap: 18px;
    font-size: 0.95em;
    color: var(--text-strong);
}
.gpu-pill {
    background: rgba(37, 99, 235, 0.15);
    color: var(--text-strong);
    padding: 6px 14px;
    border-radius: 12px;
    font-size: 0.78em;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.dark .gpu-pill {
    background: rgba(96, 165, 250, 0.25);
    color: #f8fafc;
}
.gpu-copy {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.gpu-title {
    font-size: 1em;
    font-weight: 600;
}
.gpu-meta {
    color: var(--text-subtle);
    font-size: 0.92em;
}
.gpu-inline {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    font-size: 0.92em;
    color: var(--text-subtle);
}
.gpu-inline span {
    padding-left: 10px;
    border-left: 2px solid rgba(148, 163, 184, 0.4);
}
.gpu-inline span:first-child {
    padding-left: 0;
    border-left: none;
}

.error-box {
    text-align: center;
    color: #ef4444;
    padding: 30px;
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 12px;
}
.dark .error-box {
    background: rgba(239, 68, 68, 0.15);
}
.empty-box {
    text-align: center;
    color: var(--text-subtle);
    padding: 40px;
    background: var(--surface-alt);
    border-radius: 12px;
    border: 1px dashed var(--border-elevated);
}

.icp-record {
    margin-top: 22px;
    text-align: center;
    font-size: 0.9em;
    color: var(--text-subtle);
}
.icp-record a {
    color: inherit;
    text-decoration: none;
    border-bottom: 1px dotted currentColor;
    padding-bottom: 2px;
}

/* Example grids */
#prompt-examples .gallery,
#image-examples .gallery {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    align-items: stretch;
}
#prompt-examples .gallery .gallery-item,
#image-examples .gallery .gallery-item {
    width: 100%;
    display: flex;
    align-items: stretch;
}
#prompt-examples .gallery .gallery-item > *,
#image-examples .gallery .gallery-item > * {
    width: 100%;
}
#prompt-examples .gallery .gallery-item > div,
#image-examples .gallery .gallery-item > div {
    min-width: 0 !important;
    width: 100%;
    display: block;
    padding: 12px 14px;
    border: 1px solid var(--border-elevated);
    border-radius: 10px;
    background: var(--surface-card);
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}
#prompt-examples .gallery .gallery-item:hover > div,
#image-examples .gallery .gallery-item:hover > div {
    border-color: rgba(37, 99, 235, 0.55);
    box-shadow: 0 12px 25px rgba(15, 23, 42, 0.25);
    transform: translateY(-3px);
}
"""

TIPS_HTML = """
<div class="tips-box">
    <h3>💡 Prompting Guide & Best Practices</h3>
    
    <div class="red-box">
        <strong>🚨 CRITICAL: Tips That WILL Improve Your Results</strong>
        <ul style="margin: 8px 0 0 0; padding-left: 20px;">
            <li>
                <strong>Generate 4-8 candidates and pick the best one!</strong> Results vary significantly between generations - this is NORMAL!
            </li>
            <li>
                <strong>Use GEOMETRIC descriptions:</strong> "triangular roof", "circular head", "rectangular body", "curved tail"
            </li>
            <li>
                <strong>ALWAYS specify colors for EACH element:</strong> "black outline", "red roof", "blue shirt", "green grass"
            </li>
            <li>
                <strong>Describe position & orientation:</strong> "centrally positioned", "pointing upward", "facing right", "at the bottom"
            </li>
            <li>
                <strong>Keep it SIMPLE:</strong> Avoid complex sentences. Use short, clear phrases connected by commas.
            </li>
        </ul>
    </div>

    <div class="prompt-structure">
        <strong>🧩 Recommended Prompt Structure</strong>
        <div class="prompt-example">
            [Subject] + [Shape descriptions with colors] + [Position/orientation] + [Style]
        </div>
        <p class="tip-note">
            Example: "A fox logo: triangular orange head, pointed ears, white chest marking, facing right. Minimalist flat style, centered."
        </p>
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin-top: 15px;">
        <div class="tip-category icons">
            <h4>🎯 Icons & Simple Shapes</h4>
            <p>Use clear geometric descriptions with explicit colors.</p>
            <div class="example-prompt">
                "A black triangle pointing downward, centrally positioned."
            </div>
            <div class="example-prompt">
                "A red heart shape with smooth curved edges, centered."
            </div>
            <p><strong>Keywords:</strong> <code>triangle</code> <code>circle</code> <code>arrow</code> <code>heart</code> <code>star</code> <code>centered</code></p>
        </div>
        
        <div class="tip-category animals">
            <h4>🐾 Animals</h4>
            <p>Describe as geometric shapes: oval body, round head, triangular ears, curved tail.</p>
            <div class="example-prompt">
                "Cute cat: orange round head with two triangular ears, oval orange body, curved tail. Simple cartoon style with black outlines, sitting pose."
            </div>
            <div class="example-prompt">
                "Simple black bird: oval body, small round head, pointed triangular beak facing right, triangular tail, two stick legs. Silhouette style."
            </div>
        </div>
        
        <div class="tip-category objects">
            <h4>🏠 Buildings & Objects</h4>
            <p>Use basic shapes: rectangles for walls, triangles for roofs, squares for windows.</p>
            <div class="example-prompt">
                "Simple house: red triangular roof on top, beige rectangular wall, brown rectangular door in center, two small blue square windows. Green ground at bottom."
            </div>
            <div class="example-prompt">
                "Coffee mug: brown cylindrical cup shape with curved handle on right side, three wavy steam lines rising from top. Simple flat style."
            </div>
        </div>
    </div>

    <div class="blue-box">
        <strong>🧠 Model Selection Guide</strong>
        <ul style="margin: 8px 0 0 0; padding-left: 20px;">
            <li><strong>8B Model:</strong> Higher quality, more details, better for complex illustrations. Requires more VRAM (~16GB+).</li>
            <li><strong>4B Model:</strong> Faster, less VRAM required (~8GB+). Good for simple icons and basic shapes.</li>
            <li><strong>Note:</strong> First generation with a new model size may take longer to load.</li>
        </ul>
    </div>
    
    <div class="green-box" style="margin-top: 15px;">
        <strong>🛠️ Quick Troubleshooting</strong>
        <ul style="margin: 8px 0 0 0; padding-left: 20px;">
            <li><strong>Messy/chaotic?</strong> Lower temperature to 0.3-0.4, simplify description, reduce top_k</li>
            <li><strong>Too simple/empty?</strong> Raise temperature to 0.5-0.6, add more shape details</li>
            <li><strong>Wrong colors?</strong> Explicitly name EVERY color: "red roof", "blue shirt", "black outline"</li>
            <li><strong>Missing elements?</strong> Add position words: "at top", "in center", "at bottom left"</li>
            <li><strong>Repetitive patterns?</strong> Increase repetition_penalty to 1.08-1.15</li>
            <li><strong>Inconsistent?</strong> Generate MORE candidates (6-8) and pick the best!</li>
        </ul>
    </div>
    
</div>
"""

IMAGE_TIPS_HTML = """
<div class="red-box">
    <strong>🖼️ Image-to-SVG Tips</strong>
    <ul style="margin: 8px 0 0 0; padding-left: 20px;">
        <li><strong>Best input:</strong> Simple images with clean background</li>
        <li><strong>PNG with transparency (RGBA) works best!</strong> We auto-convert to white background.</li>
        <li><strong>For complex backgrounds:</strong> Enable "Replace Background" option below.</li>
        <li><strong>Lower temperature (0.2-0.4)</strong> for more accurate reproduction.</li>
        <li><strong>Generate 4-8 candidates!</strong> Pick the one that best matches your input.</li>
    </ul>
</div>
"""

DEFAULT_EXAMPLE_PROMPTS = [
    "A black triangle pointing downward, centrally positioned.",
    "A red heart shape with smooth curved edges, centered.",
    "A yellow star with five sharp points, simple geometric design, flat color.",
    "A blue arrow pointing to the right, thick solid shape, centered.",
    "A green circle with a white checkmark inside, centered.",
    "A black plus sign with equal length arms, thick lines, centered.",
    "A simple person standing: round beige head, rectangular blue shirt body, two dark gray rectangular legs, arms at sides. Flat colors.",
    "A girl with long black hair, wearing pink dress with triangular skirt, small circular face with dot eyes and curved smile. Simple cartoon style.",
    "A child waving: large round head with brown messy hair, big circular eyes, small body in red t-shirt and blue shorts, one arm raised. Cheerful cartoon style.",
    "A person sitting on chair: side view, round head, rectangular torso in green sweater, bent legs on simple chair shape. Relaxed pose.",
    "A running person: side view silhouette in black, dynamic pose with one leg forward, arms pumping. Motion style.",
    "Circular avatar: person with short black hair, round face with two dot eyes and small curved smile, wearing blue collar shirt. Minimal style, centered in circle.",
    "Female avatar: oval face with long wavy brown hair, simple eyes, pink lips, wearing v-neck purple top. Soft cartoon style in circular frame.",
    "Profile silhouette avatar: black side view of head with short hair and glasses outline, facing right. Simple solid shape.",
    "Cute cartoon avatar: round face with big sparkly eyes, rosy cheeks, short bob haircut in orange. Kawaii style, circular frame.",
    "Professional headshot avatar: person with neat hair, neutral expression, wearing suit collar. Corporate minimal style, circular frame.",
    "Layered mountain landscape: light blue sky at top, gray triangular snow-capped mountains in middle, dark green triangular pine trees at bottom. Flat colors.",
    "Sunset beach scene: orange gradient sky at top, yellow semicircle sun on horizon, dark blue wavy ocean, tan beach strip at bottom. Simple shapes.",
    "Forest scene: light blue sky, row of 5 dark green triangular pine trees of varying heights on brown trunks, light green grass at bottom.",
    "City skyline at dusk: purple-orange gradient sky, row of black rectangular building silhouettes of different heights, some with yellow window squares.",
    "Desert landscape: light orange sky with white circle sun, tan sand dunes as curved shapes, one green cactus with arms on the right side.",
    "Countryside scene: blue sky with white fluffy clouds, green rolling hills, small red barn with white door in the center, yellow hay bales.",
    "Cute orange cat sitting: round head with two triangular ears, oval body, curved tail. Black outline cartoon style, facing forward.",
    "Simple black bird: oval body, round head, pointed triangular beak facing right, triangular tail, two stick legs. Silhouette style.",
    "Friendly cartoon dog: brown oval body, round head with floppy ears, black dot nose, wagging curved tail, four short legs. Sitting pose.",
    "Red fox logo: triangular orange face with pointed ears, white chest marking, bushy tail. Minimalist style, facing right, centered.",
    "Simple house icon: red triangular roof, beige rectangular walls, brown door in center, two blue square windows, green ground at bottom.",
    "Coffee mug: brown cylindrical cup with curved handle on right, three wavy steam lines rising from top. Flat style.",
    "Open book: two rectangular white pages spread open, black text lines on each page, brown spine in center. Simple top-down view."
]


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
        texts = [[t] for t in DEFAULT_EXAMPLE_PROMPTS]
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
    """Generate HTML for the gallery grid using CSS classes instead of inline styles."""
    if not candidates:
        return '<div class="empty-box">No candidates generated yet.</div>'
    
    items = []
    for cand in candidates:
        svg_str = cand.get("svg", "")
        # Ensure viewBox exists for proper scaling
        if svg_str and "viewBox" not in svg_str:
            svg_str = svg_str.replace("<svg", f'<svg viewBox="0 0 {TARGET_IMAGE_SIZE} {TARGET_IMAGE_SIZE}"', 1)
        
        items.append(
            f"""
            <div class="svg-card">
                <div class="svg-preview-box">
                    {svg_str}
                </div>
                <div class="svg-meta">
                    #{cand.get("index", '?')} | {cand.get("path_count", '?')} paths
                </div>
            </div>
            """
        )
    
    return f"""
    <div class="svg-gallery-grid">
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
    status = (result.get("status") or "unknown").lower()
    elapsed = result.get("elapsed_ms")
    model_size = result.get("model_size") or DEFAULT_MODEL_SIZE
    num = result.get("num_candidates", 0)
    message = result.get("message")
    job_status = result.get("job_status")
    
    # Simple icons
    icon = '✅' if status == 'ok' else '⚠️'
    base = f"{icon} {status.upper()} | Model: {model_size} | Qty: {num}"
    if elapsed:
        base += f" | Time: {elapsed}ms"
    if job_status and job_status.lower() != status:
        base += f" | Job: {job_status}"
    if message and status != "ok":
        base += f" | {message}"
    return base


def error_panel(message: str) -> str:
    return f'<div class="error-box">{message}</div>'


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
        # User feedback during generation is handled by Gradio's queue
        result = run_async(
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
            timeout_s=RUNPOD_POLL_TIMEOUT_S,
            base_interval_s=RUNPOD_POLL_BASE_INTERVAL_S,
        )
        gallery = create_gallery_html(result.get("candidates"))
        svg_code = format_svg_code(result.get("candidates"))
        status_msg = format_status(result)
        if not result.get("candidates"):
            gallery = error_panel("No valid SVG generated. Try adjusting parameters or rephrasing.")
        return gallery, svg_code, status_msg
    except RunpodClientError as e:
        msg = f"Runpod Error: {e.message}"
        return error_panel(msg), "<!-- error -->", msg
    except Exception as e:
        msg = f"System Error: {str(e)}"
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
        result = run_async(
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
            timeout_s=RUNPOD_POLL_TIMEOUT_S,
            base_interval_s=RUNPOD_POLL_BASE_INTERVAL_S,
        )
        gallery = create_gallery_html(result.get("candidates"))
        svg_code = format_svg_code(result.get("candidates"))
        status_msg = format_status(result)
        processed_img = decode_png_base64(result.get("processed_input_png_base64")) or image
        if not result.get("candidates"):
            gallery = error_panel("No valid SVG generated.")
        return gallery, svg_code, processed_img, status_msg
    except RunpodClientError as e:
        msg = f"Runpod Error: {e.message}"
        return error_panel(msg), "<!-- error -->", None, msg
    except Exception as e:
        msg = f"System Error: {str(e)}"
        return error_panel(msg), "<!-- error -->", None, msg


def build_ui():
    env_err = ensure_env_ready()
    example_texts = get_example_texts()
    example_images = get_example_images()
    text_example_count = len(example_texts)
    image_example_count = len(example_images)

    # 使用 Soft 主题作为基础，它比默认主题更现代、更圆润
    # 并允许 primary_hue 调整主色调
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )
    dark_js = """
    function() {
        const ensureDark = () => {
            document.body.classList.add('dark');
        };
        const removeExampleIframes = () => {
            document
                .querySelectorAll('#prompt-examples iframe, #image-examples iframe')
                .forEach((el) => el.remove());
        };
        const hydrateExampleTitles = () => {
            document
                .querySelectorAll('#prompt-examples .gallery-item, #image-examples .gallery-item')
                .forEach((btn) => {
                    const textBlock = btn.querySelector('div');
                    if (!textBlock) return;
                    const text = textBlock.textContent?.trim();
                    if (text) {
                        btn.setAttribute('title', text);
                        textBlock.setAttribute('title', text);
                    }
                });
        };
        const refresh = () => {
            ensureDark();
            removeExampleIframes();
            hydrateExampleTitles();
        };
        refresh();
        const observer = new MutationObserver(refresh);
        observer.observe(document.body, { subtree: true, childList: true });
    }
    """

    with gr.Blocks(title="OmniSVG Studio", css=CUSTOM_CSS, theme=theme,js=dark_js) as demo:
        gr.HTML(
            """
            <script>
            (function() {
                const root = document.documentElement;
                root.classList.add('dark');
                root.classList.remove('light');
                if (document.body) {
                    document.body.classList.add('dark');
                    document.body.classList.remove('light');
                }
            })();
            </script>
            """,
            elem_id="force-dark-mode",
        )
        # Header
        gr.HTML(
            """
            <div class="header-container">
                <h1>Fudan & Stepfun SVG Engine - OmniSVG</h1>
                <div class="header-logos">
                        <img src="https://www.fudan.edu.cn/_upload/site/00/02/2/logo.png" alt="Fudan University Logo" loading="lazy" />
                        <svg xmlns="http://www.w3.org/2000/svg" width="112" height="26" viewBox="0 0 168 40" fill="none"><g clip-path="url(#clip0_955_14525)"><path d="M99.8951 24.261L100.108 23.7769L100.335 24.2535C102.036 27.806 104.465 30.8594 107.558 33.3278L109.246 31.1771C107.452 29.7626 105.873 28.0506 104.549 26.0865C103.286 24.2157 102.297 22.2036 101.611 20.1059L101.507 19.7907H108.778V17.05H101.386V10.2979L101.606 10.2777C103.995 10.0684 106.067 9.73815 107.768 9.29439L106.923 6.67725C105.336 7.0731 103.666 7.35044 101.96 7.50677C100.247 7.66309 98.311 7.75386 96.2054 7.77403H93.1282V10.4845H96.2029C96.709 10.4845 97.4531 10.4668 98.4122 10.4315L98.6627 10.4214V17.05H91.6477V7.05292H81.2822V17.1584H85.6096V28.6129L84.0026 28.9457V20.3252H81.2822V29.5054L80.3281 29.7021L80.8798 32.357L91.6123 30.1408L91.0606 27.4858L88.3301 28.0506V23.2701H91.8831V20.5597H88.3301V17.1584H91.9008V19.7907H98.4476L98.3894 20.0781C97.7011 23.4719 95.6133 26.9614 92.1741 30.4711L92.1842 30.5139L92.121 30.5265C91.6832 30.9728 91.2226 31.4191 90.7418 31.8654L92.5233 33.953C96.0156 30.7308 98.4932 27.4682 99.8926 24.261H99.8951ZM88.9172 14.3875H84.0254V9.76588H88.9172V14.3875Z" fill="white"></path><path d="M115.638 27.1679L116.552 23.0858H124.951V25.3903H118.379L117.771 28.131H124.951V31.1465H113.138V33.857H139.512V31.1465H127.699V28.131H137.387V25.3903H127.699V23.0858H137.999V20.3451H127.699V18.2323H135.823C135.845 18.2323 135.866 18.2298 135.896 18.2247H137.217V6.80811H115.276V7.97044C115.266 8.05112 115.258 8.09651 115.258 8.14693V16.8909C115.258 16.9388 115.264 16.9842 115.274 17.0346V18.2222H116.668V18.2297H124.948V20.3426H117.162L117.354 19.4829H114.535L112.816 27.1654H115.636L115.638 27.1679ZM118.04 9.51854H134.438V11.165H118.04V9.51854ZM118.04 15.4891V13.8754H134.438V15.4891H118.04Z" fill="white"></path><path d="M64.1201 23.5878V16.8584H61.5561V23.5878C61.5561 25.088 61.2372 26.5958 60.612 28.0733C59.9869 29.5483 59.1314 30.8972 58.0684 32.0822L59.9439 33.8875C61.2372 32.4327 62.2673 30.8064 63.0039 29.0541C63.7455 27.2967 64.1201 25.4561 64.1201 23.5878Z" fill="white"></path><path d="M72.8237 16.8584H70.2598V33.6404H72.8237V16.8584Z" fill="white"></path><path d="M57.9257 24.7754C58.113 24.1551 58.2066 23.5122 58.2066 22.8667C58.2066 21.0438 57.5688 19.0797 56.3084 17.0298L56.2552 16.9441L58.3003 8.6691C58.3408 8.50521 58.3636 8.37914 58.3636 8.29342C58.3636 8.00599 58.2573 7.73368 58.0497 7.48407C57.7536 7.12856 57.4068 6.95459 56.9867 6.95459H48.9102V33.953H51.6614V9.6978H55.2656L53.4432 16.8685C53.4179 16.9466 53.4078 17.0525 53.4078 17.1962C53.4078 17.4912 53.4939 17.7711 53.671 18.056C54.0557 18.5578 54.4582 19.2915 54.8555 20.2168C55.2529 21.1447 55.4529 22.0574 55.4529 22.9323C55.4529 23.2853 55.4023 23.6509 55.2985 24.019C54.8758 25.2796 53.7647 26.2504 51.993 26.916L52.932 29.4398C55.5744 28.4843 57.255 26.916 57.9232 24.7754H57.9257Z" fill="white"></path><path d="M73.0543 11.6997C71.9256 10.255 70.7463 8.58082 69.5443 6.72259C69.4279 6.49315 69.2735 6.33179 69.0786 6.21833C68.8787 6.10487 68.6636 6.04688 68.4384 6.04688H65.9836C65.5003 6.04688 65.1308 6.24354 64.855 6.64443C63.7086 8.42954 62.5344 10.0734 61.3652 11.5283C60.196 12.9831 58.9079 14.332 57.5312 15.5347L59.1559 17.5593C60.6161 16.3717 62.0383 14.9144 63.3796 13.2251C64.6905 11.5762 65.7837 10.0684 66.629 8.74218L66.6998 8.63124H67.6842L67.7551 8.73966C70.4376 12.7788 72.91 15.7464 75.1016 17.5618L76.7566 15.5372C75.423 14.4354 74.1754 13.1444 73.0492 11.7022L73.0543 11.6997Z" fill="white"></path><path d="M168.002 12.4512H147.865V15.1919H168.002V12.4512Z" fill="white"></path><path d="M168.001 6.80811H143.082L143.074 33.8066H145.825V9.51854H168.001V6.80811Z" fill="white"></path><path d="M154.288 33.8241V31.086H151.993V20.5518H168.002V17.8389H147.865V20.5518H149.272V33.8065L154.288 33.8241Z" fill="white"></path><path d="M161.924 28.6107L165.791 33.8248L168.003 32.196L164.161 27.0146L161.924 28.6107Z" fill="white"></path><path d="M150.029 33.7262L150.098 33.8245H154.617L161.923 28.6104L160.289 26.4067L150.029 33.7262Z" fill="white"></path><path d="M168.003 24.2762L166.4 22.0474L162.525 24.8107L164.16 27.0144L168.003 24.2762Z" fill="white"></path><path d="M162.526 24.8105L160.289 26.4066L161.924 28.6102L164.162 27.0142L162.526 24.8105Z" fill="white"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M36.0391 0.5H37.6083V2.06323H36.0391V0.5ZM36.0399 2.06348H37.6083V3.6267H36.0399H36.0391H34.4707V2.06348H36.0391H36.0399ZM4.24219 21.0112V3.62663H5.83421L5.83421 21.0112H4.24219ZM21.2617 22.123H39.1079V23.6737H29.0081V39.5001H21.2617V22.123ZM9.16097 6.03189L9.15591 26.4875H0V34.0061H16.9655V13.5127H34.0297V6.03189H9.16097ZM36.0391 5.19214V3.62916H37.6083V5.19214V5.19238V6.75537H36.0391V5.19238V5.19214ZM39.1806 2.06348H37.6113V3.6267H39.1806V2.06348ZM34.4696 2.06348H32.9004V3.6267H34.4696V2.06348Z" fill="white"></path></g><defs><clipPath id="clip0_955_14525"><rect width="168" height="39" fill="white" transform="translate(0 0.5)"></rect></clipPath></defs></svg>
                </div>
                <p>Professional Vector Generation via Runpod (4B/8B Models)</p>
            </div>
            """
        )
        
        if env_err:
            gr.HTML(
                f"""
                <div class="error-box" style="padding: 15px; margin-bottom: 20px;">
                    ⚠️ Configuration Error: {env_err}
                </div>
                """
            )

        with gr.Tabs():
            # --- TAB 1: Text to SVG ---
            with gr.TabItem("📝 Text to SVG", id="tab_txt"):
                with gr.Row(equal_height=False):
                    # Left Column: Inputs
                    with gr.Column(scale=1, min_width=800):
                        text_input = gr.Textbox(
                            label="Prompt / Description",
                            placeholder="e.g. A geometric minimalist logo of a blue whale...",
                            lines=4,
                            elem_classes=["input-box"],
                            render=False,
                        )
                        if example_texts:
                            gr.Examples(
                                examples=example_texts,
                                inputs=[text_input],
                                label=f"Example Prompts ({text_example_count})",
                                examples_per_page=6,
                                elem_id="prompt-examples",
                            )
                        text_input.render()
                        
                        with gr.Group(elem_classes=["settings-group"]):
                            gr.Markdown("### ⚙️ Generation Settings")
                            text_model = gr.Dropdown(
                                choices=AVAILABLE_MODEL_SIZES,
                                value=DEFAULT_MODEL_SIZE,
                                label="Model Size",
                                info="8B: Better Quality cold start 90-150s · warm ~20s | 4B: Fastercold start 60-90s · warm ~10s"
                            )

                            text_num_candidates = gr.Slider(
                                1,
                                MAX_NUM_CANDIDATES,
                                value=min(6, MAX_NUM_CANDIDATES),
                                step=1,
                                label="Number of Candidates",
                                info="Generate 4-8 candidates and pick the best"
                            )
                            text_max_length = gr.Slider(
                                MIN_MAX_LENGTH,
                                MAX_MAX_LENGTH,
                                value=min(MAX_LENGTH, MAX_MAX_LENGTH),
                                step=64,
                                label="Max Token Length",
                                info="Lower = faster & simpler | Higher = slower & more detailed"
                            )
                            
                            with gr.Accordion("Advanced Parameters", open=False):
                                text_temperature = gr.Slider(
                                    0.1,
                                    1.0,
                                    value=TASK_CONFIGS["text-to-svg-icon"].get("default_temperature", 0.5),
                                    step=0.05,
                                    label="Temperature",
                                    info="Icons: 0.3-0.5 · Complex scenes: 0.5-0.7"
                                )
                                text_top_p = gr.Slider(
                                    0.5,
                                    1.0,
                                    value=TASK_CONFIGS["text-to-svg-icon"].get("default_top_p", 0.9),
                                    step=0.02,
                                    label="Top-P"
                                )
                                text_top_k = gr.Slider(
                                    10,
                                    100,
                                    value=TASK_CONFIGS["text-to-svg-icon"].get("default_top_k", 60),
                                    step=5,
                                    label="Top-K"
                                )
                                text_rep_penalty = gr.Slider(
                                    1.0,
                                    1.5,
                                    value=TASK_CONFIGS["text-to-svg-icon"].get("default_repetition_penalty", 1.03),
                                    step=0.01,
                                    label="Repetition Penalty",
                                    info="Increase (1.08-1.15) if outputs repeat"
                                )

                        text_button = gr.Button("✨ Generate SVG", variant="primary", size="lg")
                        text_status = gr.Textbox(
                            label="Model Status",
                            value="Ready (model loads on first generation)",
                            interactive=False,
                            max_lines=1
                        )
                        


                    # Right Column: Output
                    with gr.Column(scale=2):
                        text_gallery = gr.HTML(
                            value='<div class="empty-box">Generated SVGs will appear here once you run generation.</div>',
                            label="Gallery"
                        )
                        text_svg_code = gr.Code(
                            label="SVG Source Code", 
                            language="html", 
                            lines=12, 
                            elem_classes=["code-output"],
                            interactive=False
                        )

                text_button.click(
                    fn=handle_text_submit,
                    inputs=[text_input, text_model, text_num_candidates, text_max_length, text_temperature, text_top_p, text_top_k, text_rep_penalty],
                    outputs=[text_gallery, text_svg_code, text_status],
                    queue=True,
                )

            # --- TAB 2: Image to SVG ---
            with gr.TabItem("🖼️ Image to SVG", id="tab_img"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=800):
                        image_input = gr.Image(
                            label="Upload Reference Image",
                            type="pil",
                            image_mode="RGBA",
                            height=250,
                            sources=["upload", "clipboard"],
                            elem_classes=["input-image"],
                        )
                        with gr.Group(elem_classes=["settings-group"]):
                            gr.Markdown("### ⚙️ Generation Settings")
                            img_model = gr.Dropdown(choices=AVAILABLE_MODEL_SIZES, value=DEFAULT_MODEL_SIZE, label="Model Size")
                            img_num_candidates = gr.Slider(
                                1,
                                MAX_NUM_CANDIDATES,
                                value=DEFAULT_NUM_CANDIDATES,
                                step=1,
                                label="Number of Candidates",
                                info="More candidates = better odds of a great match"
                            )
                            img_max_length = gr.Slider(
                                MIN_MAX_LENGTH,
                                MAX_MAX_LENGTH,
                                value=min(MAX_LENGTH, MAX_MAX_LENGTH),
                                step=64,
                                label="Max Token Length",
                                info="Lower = faster & simpler | Higher = slower & detailed"
                            )
                            
                            img_replace_bg = gr.Checkbox(
                                label="Auto-remove Background",
                                value=True,
                                info="Enable if your upload has a colored background"
                            )
                            
                            with gr.Accordion("Advanced Parameters", open=False):
                                img_temperature = gr.Slider(
                                    0.1,
                                    1.0,
                                    value=TASK_CONFIGS["image-to-svg"].get("default_temperature", 0.3),
                                    step=0.05,
                                    label="Temperature (Lower = accurate)",
                                    info="0.2-0.4 works best for faithful reproductions"
                                )
                                img_top_p = gr.Slider(0.5, 1.0, value=TASK_CONFIGS["image-to-svg"].get("default_top_p", 0.9), step=0.02, label="Top-P")
                                img_top_k = gr.Slider(10, 100, value=TASK_CONFIGS["image-to-svg"].get("default_top_k", 50), step=5, label="Top-K")
                                img_rep_penalty = gr.Slider(1.0, 1.5, value=TASK_CONFIGS["image-to-svg"].get("default_repetition_penalty", 1.05), step=0.01, label="Repetition Penalty")

                        image_button = gr.Button("✨ Generate from Image", variant="primary", size="lg")
                        image_status = gr.Textbox(
                            label="Model Status",
                            value="Ready (model loads on first generation)",
                            interactive=False,
                            max_lines=1
                        )
                        
                        if example_images:
                            gr.Examples(
                                examples=example_images,
                                inputs=[image_input],
                                label=f"Example Images ({image_example_count})",
                                examples_per_page=6,
                                elem_id="image-examples",
                            )

                    with gr.Column(scale=2, min_width=500):
                        gr.HTML(IMAGE_TIPS_HTML)
                        image_processed = gr.Image(label="Processed Input Preview", type="pil", height=150, interactive=False, show_download_button=False)
                        image_gallery = gr.HTML(
                            value='<div class="empty-box">Generated SVGs will appear here after generation finishes.</div>',
                            label="Gallery"
                        )
                        image_svg_code = gr.Code(label="SVG Source Code", language="html", lines=10, elem_classes=["code-output"])

                image_button.click(
                    fn=handle_image_submit,
                    inputs=[image_input, img_model, img_num_candidates, img_max_length, img_temperature, img_top_p, img_top_k, img_rep_penalty, img_replace_bg],
                    outputs=[image_gallery, image_svg_code, image_processed, image_status],
                    queue=True,
                )

        gr.HTML(TIPS_HTML)
        gr.HTML(
            """
            <div class="icp-record">
                <span>ICP备案: <a href="https://beian.miit.gov.cn/" target="_blank" rel="noopener">渝ICP备2022010349号-2</a></span>
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniSVG Runpod Client UI")
    parser.add_argument("--listen", default="0.0.0.0", help="Listen address")
    parser.add_argument("--port", type=int, default=7862, help="Port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share")
    parser.add_argument("--debug", action="store_true", help="Show Gradio errors")
    args = parser.parse_args()
    
    app = build_ui()
    app.launch(
        server_name=args.listen, 
        server_port=args.port, 
        share=args.share, 
        show_error=args.debug,
        allowed_paths=["."] # Allow loading local examples if needed
    )

import os
import sys
import uuid
import shutil
import subprocess
import importlib.util
from pathlib import Path
from typing import Generator, Tuple, Optional

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download

# ==============================================================================
# PickStyle • HF Spaces • ZeroGPU (H200) — Stable + Streaming Logs
# Gradio: 6.5.1
# Python: 3.12
# ==============================================================================

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WAN_REPO = "Wan-AI/Wan2.1-VACE-14B"
PICKSTYLE_REPO = "Pickford/PickStyle"

VACE_SCRIPT = ROOT / "vace" / "vace_wan_inference.py"


# -----------------------
# Helpers
# -----------------------
def tail(text: str, max_chars: int = 16000) -> str:
    if not text:
        return ""
    return text[-max_chars:]


def hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HF_HUB_TOKEN")


def _snapshot(repo_id: str, local_dir: Path) -> None:
    """
    snapshot_download with token (if available), and a retry that clears partial folder.
    """
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(repo_id=repo_id, local_dir=str(local_dir), ignore_patterns=["*.git/*"])
    token = hf_token()
    if token:
        kwargs["token"] = token

    try:
        snapshot_download(**kwargs)
    except Exception as e:
        print(f"⚠️ snapshot_download failed for {repo_id}: {e}", flush=True)
        print("🔁 Retrying once after cleaning partial directory...", flush=True)
        try:
            if local_dir.exists():
                shutil.rmtree(local_dir, ignore_errors=True)
            snapshot_download(**kwargs)
        except Exception as e2:
            raise RuntimeError(f"snapshot_download failed for {repo_id} (retry also failed): {e2}") from e2


def ensure_models() -> Tuple[Path, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    wan_local = MODELS_DIR / "Wan2.1-VACE-14B"
    pick_local = MODELS_DIR / "PickStyle"

    if not wan_local.exists() or not any(wan_local.iterdir()):
        print("⬇️ Downloading WAN checkpoint (first run can be huge)...", flush=True)
        _snapshot(WAN_REPO, wan_local)

    if not pick_local.exists() or not any(pick_local.iterdir()):
        print("⬇️ Downloading PickStyle LoRA...", flush=True)
        _snapshot(PICKSTYLE_REPO, pick_local)

    lora_path = pick_local / "pickstyle_lora.pth"
    if not lora_path.exists():
        raise FileNotFoundError(f"pickstyle_lora.pth not found at: {lora_path}")

    return wan_local, lora_path


def has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def ensure_flash_attn() -> None:
    """
    Wan/VACE inference path asserts FLASH_ATTN_2_AVAILABLE.
    Installing at runtime avoids build-time failures.
    """
    if has_flash_attn():
        return

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir",
        "--no-build-isolation",
        "flash-attn",
    ]
    print("🧩 Installing flash-attn at runtime:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = tail(proc.stdout)
    print(out, flush=True)

    if proc.returncode != 0 or not has_flash_attn():
        raise RuntimeError(
            "❌ Failed to install flash-attn at runtime.\n"
            "If this keeps failing, we can pin a specific flash-attn version.\n\n"
            + out
        )


def env_summary() -> str:
    try:
        cuda_ok = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        mem = (torch.cuda.get_device_properties(0).total_memory / 1024**3) if cuda_ok else 0.0
        return f"torch={torch.__version__} | cuda={cuda_ok} | gpu={name} | vram={mem:.2f}GB"
    except Exception as e:
        return f"env_check_failed: {e}"


def _normalize_video_input(video) -> str:
    """
    Gradio 6.x Video usually returns filepath str.
    Keep a defensive fallback for dict-like shapes.
    """
    if video is None:
        return ""
    if isinstance(video, str):
        return video
    if isinstance(video, dict) and "video" in video:
        return video["video"]
    return str(video)


# -----------------------
# Non-GPU: predownload button
# -----------------------
def predownload_models() -> str:
    """
    Runs on CPU. Downloads weights to persistent storage.
    Helps avoid first-run "Run" waiting long.
    """
    wan_local = MODELS_DIR / "Wan2.1-VACE-14B"
    pick_local = MODELS_DIR / "PickStyle"
    msgs = []

    if wan_local.exists() and any(wan_local.iterdir()):
        msgs.append("✅ WAN checkpoint already exists.")
    else:
        msgs.append("⬇️ Downloading WAN checkpoint...")
        _snapshot(WAN_REPO, wan_local)
        msgs.append("✅ WAN downloaded.")

    if pick_local.exists() and any(pick_local.iterdir()):
        msgs.append("✅ PickStyle LoRA already exists.")
    else:
        msgs.append("⬇️ Downloading PickStyle LoRA...")
        _snapshot(PICKSTYLE_REPO, pick_local)
        msgs.append("✅ LoRA downloaded.")

    return "\n".join(msgs)


# -----------------------
# GPU task (stream logs)
# -----------------------
@spaces.GPU
def run_pickstyle_stream(
    video,
    prompt: str,
    adapter_name: str,
    sample_steps: int,
    skip: int,
    t_guide: float,
    c_guide: float,
    lora_rank: int,
    lora_alpha: int,
    enable_teacache: bool,
    use_ret_steps: bool,
) -> Generator[Tuple[Optional[str], str], None, None]:
    """
    Stream logs back to Gradio while running subprocess.
    Yields (output_video_path_or_None, logs_text).
    """
    video_path = _normalize_video_input(video)
    if not video_path:
        yield None, "❌ Please upload a video first."
        return

    # Confirm GPU
    log_buf = []
    log_buf.append("✅ GPU runtime: " + env_summary())
    yield None, "\n".join(log_buf)

    # Ensure flash-attn
    try:
        ensure_flash_attn()
        log_buf.append("✅ flash-attn ready.")
        yield None, "\n".join(log_buf)
    except Exception as e:
        log_buf.append("❌ flash-attn install failed:")
        log_buf.append(str(e))
        yield None, "\n".join(log_buf)
        return

    # Ensure models
    try:
        wan_dir, lora_path = ensure_models()
        log_buf.append("✅ Models ready.")
        yield None, "\n".join(log_buf)
    except Exception as e:
        log_buf.append("❌ Model download/prepare failed:")
        log_buf.append(str(e))
        yield None, "\n".join(log_buf)
        return

    # Script path
    if not VACE_SCRIPT.exists():
        log_buf.append(f"❌ Missing script: {VACE_SCRIPT}")
        log_buf.append("Make sure PickStyle repo is vendored so ./vace/vace_wan_inference.py exists.")
        yield None, "\n".join(log_buf)
        return

    job_id = uuid.uuid4().hex[:10]
    out_path = OUT_DIR / f"pickstyle_{job_id}.mp4"

    cmd = [
        "python3",
        str(VACE_SCRIPT),
        "--ckpt_dir", str(wan_dir),
        "--model_name", "vace-14B",
        "--pretrained_lora_path", str(lora_path),
        "--src_video", str(video_path),
        "--save_file", str(out_path),
        "--adapter_name", adapter_name,
        "--prompt", prompt,
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--t_guide", str(t_guide),
        "--c_guide", str(c_guide),
        "--sample_steps", str(sample_steps),
        "--skip", str(skip),
    ]
    if enable_teacache:
        cmd.append("--enable_teacache")
    if use_ret_steps:
        cmd.append("--use_ret_steps")

    log_buf.append("▶️ Running:")
    log_buf.append(" ".join(cmd))
    yield None, "\n".join(log_buf)

    # Stream subprocess output
    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
        universal_newlines=True,
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                log_buf.append(line)
                # Keep logs bounded
                joined = tail("\n".join(log_buf), 16000)
                yield None, joined
    finally:
        rc = proc.wait()

    final_logs = tail("\n".join(log_buf), 16000)

    if rc != 0:
        yield None, "❌ Process failed.\n\n" + final_logs
        return

    if not out_path.exists():
        yield None, "❌ Output video not found.\n\n" + final_logs
        return

    yield str(out_path), "✅ Done.\n\n" + final_logs


# -----------------------
# UI
# -----------------------
with gr.Blocks(title="PickStyle • ZeroGPU H200") as demo:
    gr.Markdown(
        "# 🏃 PickStyle on ZeroGPU (H200)\n"
        "建议第一次先点 **Predownload Models** 把 75GB 模型拉下来（会缓存）。\n\n"
        "如果下载提示 unauthenticated：在 Space Secrets 里加 `HF_TOKEN`（你 PRO 账号 token）会更快更稳。"
    )

    with gr.Row():
        inp_video = gr.Video(label="Input Video", sources=["upload"], format="mp4")
        out_video = gr.Video(label="Output Video", format="mp4", buttons=["download"])

    prompt = gr.Textbox(
        label="Prompt",
        value="Anime style. A boxer in white gloves trains inside a dimly lit gym.",
        lines=3,
    )

    with gr.Row():
        adapter_name = gr.Dropdown(label="Adapter", choices=["consistency"], value="consistency")
        sample_steps = gr.Slider(5, 40, value=20, step=1, label="sample_steps")
        skip = gr.Slider(0, 10, value=4, step=1, label="skip")

    with gr.Row():
        t_guide = gr.Slider(0, 10, value=5, step=0.5, label="t_guide")
        c_guide = gr.Slider(0, 10, value=4, step=0.5, label="c_guide")

    with gr.Row():
        lora_rank = gr.Slider(8, 256, value=128, step=8, label="lora_rank")
        lora_alpha = gr.Slider(8, 256, value=128, step=8, label="lora_alpha")

    with gr.Row():
        enable_teacache = gr.Checkbox(value=True, label="enable_teacache")
        use_ret_steps = gr.Checkbox(value=True, label="use_ret_steps")

    logs = gr.Textbox(label="Logs (streaming tail)", lines=16)

    with gr.Row():
        btn_dl = gr.Button("Predownload Models (CPU)", variant="secondary")
        btn_run = gr.Button("Run PickStyle (GPU)", variant="primary")

    btn_dl.click(fn=predownload_models, inputs=None, outputs=logs)

    # Streaming generator: outputs update progressively
    btn_run.click(
        fn=run_pickstyle_stream,
        inputs=[
            inp_video,
            prompt,
            adapter_name,
            sample_steps,
            skip,
            t_guide,
            c_guide,
            lora_rank,
            lora_alpha,
            enable_teacache,
            use_ret_steps,
        ],
        outputs=[out_video, logs],
    )

demo.queue(max_size=10).launch(ssr_mode=False)

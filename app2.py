import os
import sys
import uuid
import shutil
import subprocess
import importlib.util
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download

# ==============================================================================
# PickStyle • HF Spaces • ZeroGPU (H200) — Stable Runner
# - Gradio 6.5.1 compatible
# - Avoids build-time flash_attn issues by runtime install inside @spaces.GPU
# - Uses HF_TOKEN/HF_HUB_TOKEN if provided to speed downloads
# - Calls official vace/vace_wan_inference.py (vendored in Space repo)
# ==============================================================================

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# HF repos (from PickStyle README)
WAN_REPO = "Wan-AI/Wan2.1-VACE-14B"
PICKSTYLE_REPO = "Pickford/PickStyle"

# Vendored script path (you said you already put all PickStyle code into Space)
VACE_SCRIPT = ROOT / "vace" / "vace_wan_inference.py"


# -----------------------
# Utilities
# -----------------------
def tail(text: str, max_chars: int = 12000) -> str:
    if not text:
        return ""
    return text[-max_chars:]


def get_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HF_HUB_TOKEN")


def _snapshot_download(repo_id: str, local_dir: Path) -> None:
    """
    Wrapper: uses token if present, retries once, and prints friendly logs.
    """
    token = get_hf_token()
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.git/*"],
    )
    if token:
        kwargs["token"] = token

    try:
        snapshot_download(**kwargs)
    except Exception as e:
        # One retry after clearing partial dir (sometimes helps after interrupted downloads)
        print(f"⚠️ snapshot_download failed for {repo_id}: {e}", flush=True)
        print("🔁 Retrying once after cleaning partial directory...", flush=True)
        try:
            if local_dir.exists():
                shutil.rmtree(local_dir, ignore_errors=True)
            snapshot_download(**kwargs)
        except Exception as e2:
            raise RuntimeError(f"snapshot_download failed for {repo_id} (retry also failed): {e2}") from e2


def ensure_models() -> tuple[Path, Path]:
    """
    Download WAN model + PickStyle LoRA into ./models (cached on persistent storage).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    wan_local = MODELS_DIR / "Wan2.1-VACE-14B"
    pick_local = MODELS_DIR / "PickStyle"

    if not wan_local.exists() or not any(wan_local.iterdir()):
        print("⬇️ Downloading WAN checkpoint...", flush=True)
        _snapshot_download(WAN_REPO, wan_local)

    if not pick_local.exists() or not any(pick_local.iterdir()):
        print("⬇️ Downloading PickStyle LoRA...", flush=True)
        _snapshot_download(PICKSTYLE_REPO, pick_local)

    lora_path = pick_local / "pickstyle_lora.pth"
    if not lora_path.exists():
        raise FileNotFoundError(f"pickstyle_lora.pth not found at: {lora_path}")

    return wan_local, lora_path


def has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def ensure_flash_attn() -> None:
    """
    Wan/VACE path can hard-require FlashAttention2. Build-time install often fails on HF.
    We install at runtime (inside @spaces.GPU) with --no-build-isolation so torch is visible.
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
    print(tail(proc.stdout), flush=True)

    if proc.returncode != 0 or not has_flash_attn():
        raise RuntimeError(
            "❌ Failed to install flash-attn at runtime.\n"
            "Tips:\n"
            "1) Ensure your Space uses ZeroGPU (H200) and torch==2.5.1.\n"
            "2) If still failing, we may need to pin flash-attn version or add build deps.\n\n"
            + tail(proc.stdout)
        )


def summarize_env() -> str:
    try:
        cuda_ok = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        mem = (torch.cuda.get_device_properties(0).total_memory / 1024**3) if cuda_ok else 0.0
        return (
            f"torch={torch.__version__} | cuda={cuda_ok} | gpu={name} | vram={mem:.2f}GB"
        )
    except Exception as e:
        return f"env_check_failed: {e}"


# -----------------------
# Main GPU function
# -----------------------
@spaces.GPU
def run_pickstyle(
    video_path: str,
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
):
    if not video_path:
        raise ValueError("Please upload a video first.")

    # Confirm GPU really allocated
    print("✅ GPU runtime:", summarize_env(), flush=True)

    # Ensure flash-attn (needed by Wan/VACE path)
    ensure_flash_attn()

    # Ensure model assets
    wan_dir, lora_path = ensure_models()

    # Ensure script exists (vendored)
    if not VACE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Cannot find {VACE_SCRIPT}.\n"
            "You said you vendored PickStyle repo — please ensure ./vace/vace_wan_inference.py exists in the Space repo."
        )

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

    env = os.environ.copy()
    # If user set HF_TOKEN, pass through (already in env), helps any nested HF loads
    print("▶️ Running:", " ".join(cmd), flush=True)

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    logs = tail(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(logs)

    if not out_path.exists():
        raise RuntimeError("Output video not found. Logs:\n" + logs)

    return str(out_path), logs


# -----------------------
# UI
# -----------------------
with gr.Blocks(title="PickStyle • ZeroGPU H200") as demo:
    gr.Markdown(
        "# 🏃 PickStyle on ZeroGPU (H200)\n"
        "- Upload a video, enter a prompt, and generate a stylized video.\n"
        "- First run may download ~75GB WAN checkpoint (cached).\n"
        "- If you see a message about HF unauthenticated requests, set a secret `HF_TOKEN` in Space settings."
    )

    with gr.Row():
        inp_video = gr.Video(
            label="Input Video",
            sources=["upload"],
            format="mp4",
        )
        out_video = gr.Video(
            label="Output Video",
            format="mp4",
            buttons=["download"],
        )

    prompt = gr.Textbox(
        label="Prompt",
        value="Anime style. A boxer in white gloves trains inside a dimly lit gym.",
        lines=3,
    )

    with gr.Row():
        adapter_name = gr.Dropdown(
            label="Adapter",
            choices=["consistency"],
            value="consistency",
        )
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

    logs = gr.Textbox(label="Logs (tail)", lines=14)

    btn = gr.Button("Run PickStyle", variant="primary")
    btn.click(
        fn=run_pickstyle,
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

# Disable SSR if you don't need it (optional); SSR is experimental in logs.
demo.queue(max_size=10).launch(ssr_mode=False)

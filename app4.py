import os
import sys
import uuid
import glob
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
# PickStyle • HF Spaces • ZeroGPU (H200) — Stable + Streaming Logs + Wheel Cache
# Gradio: 6.5.1
# Python: 3.12
# ==============================================================================

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
WHEELS_DIR = ROOT / "wheels"  # ✅ cache flash-attn wheel here
OUT_DIR.mkdir(parents=True, exist_ok=True)
WHEELS_DIR.mkdir(parents=True, exist_ok=True)

WAN_REPO = "Wan-AI/Wan2.1-VACE-14B"
PICKSTYLE_REPO = "Pickford/PickStyle"
VACE_SCRIPT = ROOT / "vace" / "vace_wan_inference.py"


# -----------------------
# Helpers
# -----------------------
def tail(text: str, max_chars: int = 18000) -> str:
    if not text:
        return ""
    return text[-max_chars:]


def hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HF_HUB_TOKEN")


def env_summary() -> str:
    try:
        cuda_ok = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        mem = (torch.cuda.get_device_properties(0).total_memory / 1024**3) if cuda_ok else 0.0
        return f"torch={torch.__version__} | cuda={cuda_ok} | gpu={name} | vram={mem:.2f}GB"
    except Exception as e:
        return f"env_check_failed: {e}"


def _normalize_video_input(video) -> str:
    if video is None:
        return ""
    if isinstance(video, str):
        return video
    if isinstance(video, dict) and "video" in video:
        return video["video"]
    return str(video)


def _snapshot(repo_id: str, local_dir: Path) -> None:
    """
    snapshot_download with token (if available), retry once after cleaning partial folder.
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
        print("⬇️ Downloading WAN checkpoint (first run is huge)...", flush=True)
        _snapshot(WAN_REPO, wan_local)

    if not pick_local.exists() or not any(pick_local.iterdir()):
        print("⬇️ Downloading PickStyle LoRA...", flush=True)
        _snapshot(PICKSTYLE_REPO, pick_local)

    lora_path = pick_local / "pickstyle_lora.pth"
    if not lora_path.exists():
        raise FileNotFoundError(f"pickstyle_lora.pth not found at: {lora_path}")

    return wan_local, lora_path


# -----------------------
# flash-attn wheel caching
# -----------------------
def has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def find_cached_flash_attn_wheel() -> Optional[str]:
    wheels = sorted(glob.glob(str(WHEELS_DIR / "flash_attn-*.whl")))
    return wheels[-1] if wheels else None


def pip_install(args: list[str]) -> str:
    cmd = [sys.executable, "-m", "pip"] + args
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = tail(proc.stdout)
    print(out, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(out)
    return out


def ensure_flash_attn() -> None:
    """
    ✅ Best practice on ZeroGPU:
    - Prefer installing from cached wheel in ./wheels (fast, no rebuild)
    - If not present, build wheel once, store it, then install from it
    """
    if has_flash_attn():
        return

    # 1) Try install from cached wheel
    wheel_path = find_cached_flash_attn_wheel()
    if wheel_path:
        print(f"🧩 Installing flash-attn from cached wheel: {wheel_path}", flush=True)
        pip_install(["install", "--no-cache-dir", wheel_path])
        if has_flash_attn():
            return

    # 2) Build wheel (one-time) then install
    print("🧩 Building flash-attn wheel (one-time) ...", flush=True)
    # Build wheel into WHEELS_DIR
    pip_install([
        "wheel",
        "--no-cache-dir",
        "--no-build-isolation",
        "-w", str(WHEELS_DIR),
        "flash-attn",
    ])

    wheel_path = find_cached_flash_attn_wheel()
    if not wheel_path:
        raise RuntimeError("flash-attn wheel build finished but no wheel found in ./wheels")

    print(f"🧩 Installing flash-attn from newly built wheel: {wheel_path}", flush=True)
    pip_install(["install", "--no-cache-dir", wheel_path])

    if not has_flash_attn():
        raise RuntimeError("flash-attn install completed but import still fails.")


# -----------------------
# CPU: predownload button
# -----------------------
def predownload_models() -> str:
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
# GPU: run inference with streaming logs
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

    video_path = _normalize_video_input(video)
    log_buf: list[str] = []

    if not video_path:
        yield None, "❌ Please upload a video first."
        return

    log_buf.append("✅ GPU runtime: " + env_summary())
    yield None, "\n".join(log_buf)

    # Ensure flash-attn (cached wheel if possible)
    try:
        ensure_flash_attn()
        log_buf.append("✅ flash-attn ready (cached wheel preferred).")
        yield None, tail("\n".join(log_buf))
    except Exception as e:
        log_buf.append("❌ flash-attn failed:")
        log_buf.append(str(e))
        yield None, tail("\n".join(log_buf))
        return

    # Ensure models
    try:
        wan_dir, lora_path = ensure_models()
        log_buf.append("✅ Models ready.")
        yield None, tail("\n".join(log_buf))
    except Exception as e:
        log_buf.append("❌ Models failed:")
        log_buf.append(str(e))
        yield None, tail("\n".join(log_buf))
        return

    if not VACE_SCRIPT.exists():
        yield None, f"❌ Missing script: {VACE_SCRIPT}\nPlease ensure PickStyle repo is vendored with ./vace/vace_wan_inference.py"
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
    yield None, tail("\n".join(log_buf))

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                log_buf.append(line)
                yield None, tail("\n".join(log_buf))
    finally:
        rc = proc.wait()

    final_logs = tail("\n".join(log_buf))
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
        "✅ 已适配 ZeroGPU H200（含 flash-attn wheel 缓存 + 推理日志流式输出）。\n\n"
        "建议：在 Space 的 Secrets 里设置 `HF_TOKEN`，下载更快更稳。"
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

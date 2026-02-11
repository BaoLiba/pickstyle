import os
import uuid
import subprocess
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download

# -----------------------
# Paths
# -----------------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# HF repos (from PickStyle README)
# -----------------------
WAN_REPO = "Wan-AI/Wan2.1-VACE-14B"
PICKSTYLE_REPO = "Pickford/PickStyle"


def ensure_pickstyle_repo_local():
    """
    Ensure the PickStyle code repo exists locally.
    If you already committed the PickStyle repo into your Space, this will be skipped.
    If not, we'll snapshot_download it into ./pickstyle_repo and run the script from there.
    """
    local_repo = ROOT / "pickstyle_repo"
    if local_repo.exists() and (local_repo / "vace" / "vace_wan_inference.py").exists():
        return local_repo

    local_repo.mkdir(parents=True, exist_ok=True)

    # Use token automatically if present in env (HF_TOKEN/HF_HUB_TOKEN)
    snapshot_download(
        repo_id="PickfordAI/pickstyle",  # code repo on GitHub is PickfordAI/pickstyle
        repo_type="model",               # harmless; snapshot_download just needs repo_id
        local_dir=str(local_repo),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.git/*"],
    )

    # If above doesn't work (because it's GitHub, not HF), user should vendor repo.
    # We'll still check for the script; if missing, raise clear error.
    script_path = local_repo / "vace" / "vace_wan_inference.py"
    if not script_path.exists():
        raise FileNotFoundError(
            "Cannot find vace/vace_wan_inference.py.\n"
            "✅ Fix (recommended): put PickStyle repo files into your Space repo (so that ./vace exists).\n"
            "Then this app will run without downloading code."
        )
    return local_repo


def ensure_models():
    """
    Download WAN model + PickStyle LoRA into ./models.
    This is cached on disk between runs (within the Space persistent storage).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    wan_local = MODELS_DIR / "Wan2.1-VACE-14B"
    pick_local = MODELS_DIR / "PickStyle"

    if not wan_local.exists() or not any(wan_local.iterdir()):
        snapshot_download(
            repo_id=WAN_REPO,
            local_dir=str(wan_local),
            local_dir_use_symlinks=False,
        )

    if not pick_local.exists() or not any(pick_local.iterdir()):
        snapshot_download(
            repo_id=PICKSTYLE_REPO,
            local_dir=str(pick_local),
            local_dir_use_symlinks=False,
        )

    lora_path = pick_local / "pickstyle_lora.pth"
    if not lora_path.exists():
        raise FileNotFoundError(f"pickstyle_lora.pth not found at: {lora_path}")

    return wan_local, lora_path


def tail(text: str, max_chars: int = 8000) -> str:
    if text is None:
        return ""
    return text[-max_chars:]


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
    """
    Gradio 6.x Video input: passes a str filepath (per docs).
    """
    if not video_path:
        raise ValueError("Please upload a video first.")

    # ZeroGPU GPU is only available inside this function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Torch available device:", device)

    # Ensure local assets
    wan_dir, lora_path = ensure_models()

    # IMPORTANT:
    # Recommended: vendor the PickStyle repo into your Space so ./vace exists.
    # If you already have ./vace/vace_wan_inference.py in Space root, we use that.
    script_in_root = ROOT / "vace" / "vace_wan_inference.py"
    if script_in_root.exists():
        workdir = ROOT
        script_path = script_in_root
    else:
        # fallback attempt (if you didn't vendor it)
        workdir = ensure_pickstyle_repo_local()
        script_path = workdir / "vace" / "vace_wan_inference.py"

    job_id = uuid.uuid4().hex[:10]
    out_path = OUT_DIR / f"pickstyle_{job_id}.mp4"

    cmd = [
        "python3",
        str(script_path),
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

    print("Running:", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )

    logs = tail(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(logs)

    if not out_path.exists():
        raise RuntimeError("Output video not found. Logs:\n" + logs)

    # Gradio Video output expects a filepath (string/Path)
    return str(out_path), logs


with gr.Blocks(title="PickStyle • ZeroGPU H200") as demo:
    gr.Markdown(
        "# 🏃 PickStyle on ZeroGPU (H200)\n"
        "Upload a video, provide a prompt, and get a stylized video output (mp4)."
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
            buttons=["download"],  # Gradio 6 uses buttons instead of show_download_button
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

demo.queue(max_size=10).launch()

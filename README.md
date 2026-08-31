# OOS VLM Evaluation

This repository contains the OOS video-question-answering evaluation protocol,
lmms-eval model adapters, launchers, scoring code, and reproducibility metadata.
Model weights and external model source repositories are not redistributed.

The evaluation uses the authors' released model implementations and checkpoints
at pinned revisions. Small compatibility patches for offline loading, current
PyTorch APIs, and memory-efficient initialization are documented in
[`patches/`](patches/README.md). Evaluation-specific frame sampling, prompts,
generation settings, and scoring are implemented in this repository.

Reproducibility entry points:

- [`models/manifest.yaml`](models/manifest.yaml): external source and checkpoint revisions.
- [`models/README.md`](models/README.md): source bootstrap, login-node download, and offline verification commands.
- [`.env.example`](.env.example): portable storage/cache configuration.
- [`launchers/README.md`](launchers/README.md): model presets and execution commands.

The setup below documents the validated ETH Euler CUDA 12.8 environment. Other
systems should copy `.env.example` to `.env` and override `OOS_STORAGE_ROOT`,
environment activation, FFmpeg, and Slurm settings as needed.

## Reproducible Euler environment setup

This creates a Python 3.10 environment with CUDA 12.8 PyTorch.

### 1. Install uv

```bash
USERNAME="$(whoami)"
cd "/cluster/home/${USERNAME}"
git clone https://github.com/Zoulution/oos_vlm_evaluation.git
cd "/cluster/home/${USERNAME}/oos_vlm_evaluation"

python -m pip install --user -U uv
export PATH="$HOME/.local/bin:$PATH"

grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

uv --version
```

### 2. Create and activate the venv

This keeps the full virtual environment physically inside the repository on
home storage:

- project directory: `/cluster/home/<username>/oos_vlm_evaluation`
- project venv directory: `/cluster/home/<username>/oos_vlm_evaluation/.venv-cu128-home`

```bash
export USERNAME="$(whoami)"

cd "/cluster/home/${USERNAME}/oos_vlm_evaluation"

export OOS_VENV_DIR="/cluster/home/${USERNAME}/oos_vlm_evaluation/.venv-cu128-home"
uv venv --python 3.10 "$OOS_VENV_DIR"

source "$OOS_VENV_DIR/bin/activate"
```

### 3. Install CUDA 12.8 PyTorch

```bash
uv pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install this repo

```bash
printf "torch==2.7.1\ntorchvision==0.22.1\n" > constraints-torch.txt
uv pip install -e ".[all]" -c constraints-torch.txt
```

### 5. Verify

```bash
uv pip show torch torchvision lmms-eval --python .venv-cu128-home/bin/python
```

Expected key versions:

```text
torch 2.7.1+cu128
torchvision 0.22.1+cu128
lmms-eval 0.7.1
Editable project location: /cluster/home/<username>/oos_vlm_evaluation
```

Note: this `uv` venv may not have `python -m pip`. Use
`uv pip ... --python .venv-cu128-home/bin/python` instead.

### 6. Install ffmpeg
conda create -p /cluster/home/${USERNAME}/scratch/venvs/ffmpeg_env \
  -c conda-forge ffmpeg -y

Copy `.env.example` to `.env` and set any machine-specific storage or tool
paths there. The launcher derives paths from `OOS_STORAGE_ROOT`; do not edit or
hard-code `USER` in `launchers/oos_env.sh`.


### 7. Clean the uv cache from time to time to save space
check space left
```bash
lquota
```

remove cache safely
```bash
uv cache clean
```

## Cambrian-P setup (CUDA 12.8 and offline compute nodes)

Cambrian-P uses the CUDA 12.8 environment created above, plus a small package
overlay on scratch storage. The overlay supplies the older Transformers stack
required by the official Cambrian-P implementation without copying or modifying
the main Qwen environment.

Do **not** install `cambrian-s` for this model. Cambrian-P uses the official
[`cambrian-mllm/cambrian-p`](https://github.com/cambrian-mllm/cambrian-p)
repository and imports the `cambrianp` package.

### 8. Clone and patch the official Cambrian-P repository

The patch is pinned to official Cambrian-P commit
`b3c15527a9f9c4b70020b874eab66f67b2b03901`.

```bash
export USERNAME="$(whoami)"
export OOS_REPO_DIR="/cluster/home/${USERNAME}/oos_vlm_evaluation"
export CAMBRIAN_P_PATH="/cluster/home/${USERNAME}/scratch/cambrian-p"
export CAMBRIAN_P_VGGT_PATH="${CAMBRIAN_P_PATH}/vggt"

git clone https://github.com/cambrian-mllm/cambrian-p.git "$CAMBRIAN_P_PATH"
git -C "$CAMBRIAN_P_PATH" checkout b3c15527a9f9c4b70020b874eab66f67b2b03901
git -C "$CAMBRIAN_P_PATH" apply \
  "$OOS_REPO_DIR/patches/cambrian-p-offline.patch"
```

The patch only changes two upstream loading paths when
`CAMBRIAN_P_OFFLINE=1`:

- It uses the complete `LlavaQwenConfig` bundled with the Cambrian-P
  checkpoint instead of looking up a LLaVA-OneVision compatibility config.
- It constructs SigLIP from the bundled configuration and lets the
  Cambrian-P checkpoint load the vision-tower weights. A separate
  `google/siglip-*` download is therefore not required.

To confirm that the patch is present:

```bash
git -C "$CAMBRIAN_P_PATH" apply --reverse --check \
  "$OOS_REPO_DIR/patches/cambrian-p-offline.patch"
```

That command exits successfully when the patch has already been applied.

### 9. Create the low-space Cambrian-P overlay

```bash
export OOS_VENV_DIR="${OOS_REPO_DIR}/.venv-cu128-home"
export CAMBRIAN_OVERLAY="/cluster/home/${USERNAME}/scratch/python-overlays/cambrian-cu128"

mkdir -p "$CAMBRIAN_OVERLAY"

uv pip install \
  --python "$OOS_VENV_DIR/bin/python" \
  --target "$CAMBRIAN_OVERLAY" \
  --link-mode copy \
  --no-deps \
  transformers==4.37.0 \
  tokenizers==0.15.0 \
  huggingface-hub==0.25.2 \
  accelerate==0.23.0 \
  einops==0.6.1 \
  einops-exts==0.0.4 \
  sentencepiece==0.1.99 \
  decord==0.6.0 \
  imageio==2.37.3 \
  iopath==0.1.10 \
  portalocker==4.0.0
```

The overlay is added ahead of the base environment only for the `cambrian_p`
preset. The effective Python search order is:

```text
cambrian-p source -> cambrian-p/vggt -> Cambrian overlay -> base CUDA 12.8 environment
```

Verify the imports without loading the 7B checkpoint:

```bash
PYTHONPATH="$CAMBRIAN_P_PATH:$CAMBRIAN_P_VGGT_PATH:$CAMBRIAN_OVERLAY" \
  "$OOS_VENV_DIR/bin/python" -c \
  'import torch, decord, transformers; from cambrianp.model.builder import load_pretrained_model; print(torch.__version__, torch.version.cuda); print(transformers.__version__); print("Cambrian-P imports OK")'
```

Expected core versions are CUDA 12.8 PyTorch `2.7.1+cu128` and Transformers
`4.37.0` from the overlay.

### 10. Download the Cambrian-P checkpoint before submitting

Compute nodes run in offline mode, so download the checkpoint from a login node:

```bash
export HF_HOME="/cluster/home/${USERNAME}/scratch/hf_cache"

"$OOS_VENV_DIR/bin/hf" download nyu-visionx/Cambrian-P-7B \
  --cache-dir "$HF_HOME"
```

The launcher resolves this cached snapshot and copies it to node-local temporary
storage. Make sure the Slurm job requests enough temporary space for the roughly
16 GB checkpoint.

The default Cambrian-P paths are configured in
`launchers/oos_env.sh`. If your clone or overlay is elsewhere, export
`CAMBRIAN_P_PATH`, `CAMBRIAN_P_VGGT_PATH`, and `CAMBRIAN_OVERLAY` before calling
`sbatch`.

### 11. Run Cambrian-P

```bash
cd "$OOS_REPO_DIR"
OOS_MODEL=cambrian_p sbatch launchers/slurm_oos_eval.sh
```

For a small smoke test:

```bash
OOS_MODEL=cambrian_p OOS_LIMIT=2 sbatch launchers/slurm_oos_eval.sh
```

The Cambrian-P preset enables `CAMBRIAN_P_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`. It also supplies the official camera-token, spatial
pooling, and Qwen 1.5 conversation settings used by the local lmms-eval adapter.

## Spatial-MLLM setup (CUDA 12.8 and offline compute nodes)

Spatial-MLLM uses the same CUDA 12.8 environment created in sections 1-5, plus
a small dependency overlay on scratch storage. This avoids duplicating the full
Qwen environment and keeps Spatial-MLLM's pinned Transformers version isolated
from the other model presets.

The RTX PRO 6000/Blackwell setup uses PyTorch `2.7.1+cu128` and SDPA. Do not
install the FlashAttention wheel linked in the upstream Spatial-MLLM README: it
was built for PyTorch 2.6 and is not compatible with this environment.

### 12. Clone the official Spatial-MLLM repository

The setup below is validated against official Spatial-MLLM commit
`9fc47382c7bc5ab52951e6e2e64db08fca0948ee`. No patch to the upstream
Spatial-MLLM repository is required.

```bash
export USERNAME="$(whoami)"
export OOS_REPO_DIR="/cluster/home/${USERNAME}/oos_vlm_evaluation"
export SPATIAL_MLLM_REPO="/cluster/home/${USERNAME}/scratch/Spatial-MLLM"
export SPATIAL_MLLM_EXTERNAL_PATH="${SPATIAL_MLLM_REPO}/src/qwenvl/external"

git clone https://github.com/THU-SI/Spatial-MLLM.git "$SPATIAL_MLLM_REPO"
git -C "$SPATIAL_MLLM_REPO" checkout \
  9fc47382c7bc5ab52951e6e2e64db08fca0948ee
```

If the repository is already present, confirm its revision with:

```bash
git -C "$SPATIAL_MLLM_REPO" rev-parse HEAD
```

### 13. Create the low-space Spatial-MLLM overlay

```bash
export OOS_VENV_DIR="${OOS_REPO_DIR}/.venv-cu128-home"
export SPATIAL_OVERLAY="/cluster/home/${USERNAME}/scratch/python-overlays/spatial-mllm-cu128"

mkdir -p "$SPATIAL_OVERLAY"

uv pip install \
  --python "$OOS_VENV_DIR/bin/python" \
  --target "$SPATIAL_OVERLAY" \
  --link-mode copy \
  --no-deps \
  transformers==4.51.3 \
  tokenizers==0.21.4 \
  huggingface-hub==0.36.2 \
  decord==0.6.0 \
  deepspeed==0.19.2 \
  hjson==3.1.0 \
  ninja==1.13.0 \
  py-cpuinfo==9.0.0
```

The remaining runtime dependencies are supplied by the base environment. The
`spatial_mllm` preset in `launchers/oos_env.sh` automatically selects this base
environment and overlay. Its effective Python search order is:

```text
Spatial-MLLM source -> bundled VGGT -> Spatial overlay -> base CUDA 12.8 environment
```

Verify the environment without loading the checkpoint weights:

```bash
PYTHONPATH="$SPATIAL_MLLM_REPO:$SPATIAL_MLLM_EXTERNAL_PATH:$SPATIAL_OVERLAY" \
  "$OOS_VENV_DIR/bin/python" -c \
  'import torch, torchvision, transformers, decord, deepspeed; from transformers import Qwen2_5_VLProcessor; from src.qwenvl.model.spatial_mllm import SpatialMLLMConfig, SpatialMLLMForConditionalGeneration; print(torch.__version__, torch.version.cuda); print(torchvision.__version__, transformers.__version__, deepspeed.__version__); print("Spatial-MLLM imports OK")'
```

Expected core versions are PyTorch `2.7.1+cu128`, torchvision `0.22.1+cu128`,
Transformers `4.51.3`, and DeepSpeed `0.19.2`. An NVML warning is expected if
this verification is run on a login node without GPU access.

### 14. Download the Spatial-MLLM checkpoint before submitting

Compute nodes cannot reach Hugging Face, so download the checkpoint on a login
node into the path expected by the launcher:

```bash
export HF_HOME="/cluster/home/${USERNAME}/scratch/hf_cache"
export SPATIAL_MLLM_CKPT="/cluster/home/${USERNAME}/scratch/checkpoints/Spatial-MLLM-v1.1-Instruct-820K"

"$OOS_VENV_DIR/bin/hf" download Diankun/Spatial-MLLM-v1.1-Instruct-820K \
  --local-dir "$SPATIAL_MLLM_CKPT"
```

The launcher stages this local checkpoint to node-local temporary storage. If
the clone, overlay, or checkpoint is stored elsewhere, export
`SPATIAL_MLLM_REPO`, `SPATIAL_MLLM_EXTERNAL_PATH`, `SPATIAL_OVERLAY`, or
`SPATIAL_MLLM_CKPT` before calling `sbatch`.

### 15. Run Spatial-MLLM

```bash
cd "$OOS_REPO_DIR"
OOS_MODEL=spatial_mllm sbatch launchers/slurm_oos_eval.sh
```

For a small smoke test:

```bash
OOS_MODEL=spatial_mllm OOS_LIMIT=2 sbatch launchers/slurm_oos_eval.sh
```

## How to run jobs
1. edit the path of vqa that you want to evaluate: `/cluster/home/$USERNAME/oos_vlm_evaluation/lmms_eval/tasks/oos_videoqa/oos_videoqa_multi_turn.yaml`. (you can also change the system prompt, maximum number of output tokens, temperature etc here)

2. decide the model you want to use, and modify some model specific arguments such as model weights dir, fps, max_num_frames...etc here `/cluster/home/$USERNAME/oos_vlm_evaluation/launchers/models`.

3. if you want to modify some general argument/environment variable, such as which steps to evaluate, if videos should be fed, which evaluation mode, which env to activate, do it here `/cluster/home/$USERNAME/oos_vlm_evaluation/launchers/oos_env.sh`.

4. finally, to launch a slurm job here, you can modify the type of gpu to request, runtime request, etc here `/cluster/home/$USERNAME/oos_vlm_evaluation/launchers/slurm_oos_eval.sh`. And after that, you can lauch the slurm job by this example:
```bash
OOS_MODEL=qwen3_6 sbatch launchers/slurm_oos_eval.sh
```

The SenseNova-SI presets use the locally cached latest releases selected for
this benchmark:

```bash
OOS_MODEL=sensenova_internvl sbatch launchers/slurm_oos_eval.sh  # SenseNova-SI-1.5-InternVL3-8B
OOS_MODEL=sensenova_qwen sbatch launchers/slurm_oos_eval.sh     # SenseNova-SI-1.3-Qwen3-VL-8B
```

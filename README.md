# Reproducible Environment Setup

This creates a Python 3.10 environment with CUDA 12.8 PyTorch.

## 1. Install uv

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

## 2. Create and activate the venv

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

## 3. Install CUDA 12.8 PyTorch

```bash
uv pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

## 4. Install this repo

```bash
printf "torch==2.7.1\ntorchvision==0.22.1\n" > constraints-torch.txt
uv pip install -e ".[all]" -c constraints-torch.txt
```

## 5. Verify

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

## 6. Install ffmpeg
conda create -p /cluster/home/${USERNAME}/scratch/venvs/ffmpeg_env \
  -c conda-forge ffmpeg -y

If you follow the instructions, then to run the slurm jobs, you only need to navigate to `/cluster/home/${USERNAME}/oos_vlm_evaluation/launchers/oos_env.sh` to change the `USER`.


## 7. Clean the uv cache from time to time to save space
check space left
```bash
lquota
```

remove cache safely
```bash
uv cache clean
```

# How to run jobs
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

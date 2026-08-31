# Launchers

This folder has one generic OOS evaluation path and small model presets.

- `oos_env.sh`: committed Team 1 example environment. Edit shared paths, token, cache, and output settings here.
- `run_oos_eval.sh`: generic local runner. Select the model with `OOS_MODEL`.
- `slurm_oos_eval.sh`: generic Slurm wrapper around `run_oos_eval.sh`.
- `models/*.sh`: model-specific checkpoints, `lmms-eval` model names and arguments, output subdirectories, and small setup hooks.
- `team1/*.sh`: ETH 3dv Team 1 convenience wrappers for the GB10 environment.

## Model presets

Pass a preset name through `OOS_MODEL`. The available presets are:

| Preset | Model |
| --- | --- |
| `qwen3_6` | Qwen 3.6 |
| `qwen3_vl` | Qwen3-VL |
| `llava` | LLaVA-OneVision 1.5 |
| `internvl` | InternVL 3.5 |
| `phi4` | Phi-4 Multimodal |
| `vlm3r` | VLM-3R |
| `stream3d_vlm` | Stream3D-VLM |
| `cambrian_p` | Cambrian-P |
| `spatial_mllm` | Spatial-MLLM v1.1 |
| `sensenova_internvl` | SenseNova-SI 1.5 InternVL3 8B |
| `sensenova_qwen` | SenseNova-SI 1.3 Qwen3-VL 8B |

FFmpeg:

The OOS task uses ffmpeg for video prefix/frame extraction. Check on the machine where the run will execute:

```bash
ffmpeg -version
uname -m
```

If `ffmpeg -version` works, use it with `export FFMPEG_PATH="$(command -v ffmpeg)"`. If it is missing, install or choose a binary for the node architecture shown by `uname -m`. Do not copy an ffmpeg binary from a different architecture.

Install ffmpeg from the same architecture as the node that will run evaluation. For an interactive compute-node shell, adapt this to your cluster:

```bash
srun --time=01:00:00 --pty bash
uname -m
```

Typical choices:

```bash
# Cluster module
module avail ffmpeg
module load ffmpeg
export FFMPEG_PATH="$(command -v ffmpeg)"

# Venv fallback
uv pip install imageio-ffmpeg
export FFMPEG_PATH="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"

# Cluster-approved package manager or prebuilt binary
export FFMPEG_PREFIX=/path/to/ffmpeg-prefix-for-$(uname -m)
export FFMPEG_PATH=$FFMPEG_PREFIX/bin/ffmpeg
```

For shared-library installs, also set:

```bash
export LD_LIBRARY_PATH=$FFMPEG_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PATH=$FFMPEG_PREFIX/bin:${PATH}
```

Examples:

```bash
OOS_MODEL=qwen3_6 bash launchers/run_oos_eval.sh
OOS_MODEL=internvl OOS_LIMIT=2 bash launchers/run_oos_eval.sh
OOS_MODEL=qwen3_vl OOS_TARGET_REFERENCE_POSITION=before bash launchers/run_oos_eval.sh
OOS_MODEL=cambrian_p OOS_LIMIT=2 OOS_VENV=/cluster/home/fangma/scratch/venvs/oos_vlm_evaluation-cu124-cambrianp/bin/activate bash launchers/run_oos_eval.sh
OOS_MODEL=qwen3_6 OOS_VENV=.venv/bin/activate sbatch launchers/slurm_oos_eval.sh
OOS_MODEL=spatial_mllm OOS_VENV=.venv-SpatialMLLM/activate_oos.sh bash launchers/run_oos_eval.sh
OOS_MODEL=sensenova_internvl OOS_VENV=.venv-SenseNovaSI/activate_oos.sh bash launchers/run_oos_eval.sh
OOS_MODEL=sensenova_qwen OOS_VENV=.venv-SenseNovaSI/activate_oos.sh bash launchers/run_oos_eval.sh
```

`OOS_TARGET_REFERENCE_POSITION` controls where a dataset-provided target
reference image is placed relative to the video. Its accepted values are
`before` and `after`; the default is `after`.

The VLM-3R adapter passes that reference as a separate `image` modality beside
the sampled `video` tensor, so it does not reduce `VLM3R_FRAMES`.

Submit all model smoke tests from a login node, or select individual models:

```bash
launchers/submit_oos_smoke_tests.sh
launchers/submit_oos_smoke_tests.sh --qwen3-6
launchers/submit_oos_smoke_tests.sh --qwen3-vl --llava --limit 1
launchers/submit_oos_smoke_tests.sh --dry-run
```

`bash launchers/run_oos_eval.sh` runs the evaluation immediately on the current machine, so use it from an interactive compute-node allocation. `sbatch launchers/slurm_oos_eval.sh` submits the same evaluation as a queued Slurm job; Slurm allocates the GPU and starts the command later. For `spatial_mllm` and the two SenseNova presets, `oos_env.sh` already selects these same model-specific virtual environments by default, so the explicit `OOS_VENV=...` above is optional.

Add a new model by creating `models/<preset>.sh` that sets `MODEL`, `MODEL_ARGS`, and `OUTPUT_SUBDIR`.

## Offline model loading and node-local staging

Keep model defaults in the model preset rather than `oos_env.sh`. The Qwen3-VL
preset uses the single local checkpoint directory created by `hf download
--local-dir` and stages that directory to node-local storage.

When `OOS_STAGE_MODEL_TO_TMP=1`, the launcher infers the staging source from the
`pretrained` model argument:

- A local checkpoint directory is copied directly to the job's node-local temporary directory.
- A Hugging Face model ID such as `Qwen/Qwen3-VL-8B-Instruct` is resolved from
  `${HF_HUB_CACHE:-$HF_HOME/hub}` and then copied to node-local storage.

To use a model ID without network access, populate the standard Hub cache before
submitting the job. Do not manually create a flat directory under `hub`; let the
Hugging Face CLI create its `models--ORG--MODEL/{blobs,refs,snapshots}` layout.

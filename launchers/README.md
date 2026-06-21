# Launchers

This folder has one generic OOS evaluation path and small model presets.

- `oos_env.sh`: committed Team 1 example environment. Edit paths, token, cache, and output settings here.
- `run_oos_eval.sh`: generic local runner. Select the model with `OOS_MODEL`.
- `slurm_oos_eval.sh`: generic Slurm wrapper around `run_oos_eval.sh`.
- `models/*.sh`: model-specific `lmms-eval` model names, arguments, output subdirectories, and small setup hooks.
- `team1/*.sh`: ETH 3dv Team 1 convenience wrappers for the GB10 environment.


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
OOS_MODEL=qwen3_6 OOS_VENV=.venv/bin/activate sbatch launchers/slurm_oos_eval.sh
```

Add a new model by creating `models/<preset>.sh` that sets `MODEL`, `MODEL_ARGS`, and `OUTPUT_SUBDIR`.

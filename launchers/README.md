# Launchers

This folder has one generic OOS evaluation path and small model presets.

- `oos_env.sh`: committed Team 1 example environment. Edit paths, token, cache, and output settings here.
- `run_oos_eval.sh`: generic local runner. Select the model with `OOS_MODEL`.
- `slurm_oos_eval.sh`: generic Slurm wrapper around `run_oos_eval.sh`.
- `models/*.sh`: model-specific `lmms-eval` model names, arguments, output subdirectories, and small setup hooks.
- `team1/*.sh`: ETH 3dv Team 1 convenience wrappers for the GB10 environment.


FFmpeg:

`oos_env.sh` sets `FFMPEG_PATH`, `PATH`, and `LD_LIBRARY_PATH` for the Team 1 ffmpeg install. On another machine, change `FFMPEG_PATH` to your ffmpeg binary, or leave it unset if `ffmpeg` is already available on `PATH`.

Examples:

```bash
OOS_MODEL=qwen3_6 bash launchers/run_oos_eval.sh
OOS_MODEL=internvl OOS_LIMIT=2 bash launchers/run_oos_eval.sh
OOS_MODEL=qwen3_6 OOS_VENV=.venv/bin/activate sbatch launchers/slurm_oos_eval.sh
```

Add a new model by creating `models/<preset>.sh` that sets `MODEL`, `MODEL_ARGS`, and `OUTPUT_SUBDIR`.

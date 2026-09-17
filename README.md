# OOS VLM Evaluation

OOS video question answering with [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Setup

Requires Linux x86-64, Python 3.10+, FFmpeg, and an NVIDIA GPU with a CUDA 12.8-compatible driver.

```bash
git clone https://github.com/Zoulution/oos_vlm_evaluation.git
cd oos_vlm_evaluation

# Choose a directory with enough space for environments, checkpoints, and caches.
export OOS_STORAGE_ROOT="/absolute/path/to/storage"

# Prepare one model, or use --all to prepare every model.
bash setup.sh --model qwen3_5_9b
```

## Evaluate

The Hugging Face dataset is not published yet. Until then, supply your [local dataset](docs/dataset.md):

```bash
# Set this to your prepared questions file; referenced media must be accessible.
export OOS_DATASET_JSONL="/absolute/path/to/questions.jsonl"

# On a GPU machine: test two questions, using the downloaded checkpoints.
OOS_MODEL=qwen3_5_9b OOS_OFFLINE=1 OOS_LIMIT=2 bash launchers/run_oos_eval.sh

# Full evaluation
OOS_MODEL=qwen3_5_9b OOS_OFFLINE=1 bash launchers/run_oos_eval.sh
```

Results: `$OOS_STORAGE_ROOT/lmms-eval/outputs/oos_videoqa/<model>/`.
Questions run independently with `prefix` video context.

## All models

```bash
bash setup.sh --all
```

Presets: `qwen3_5_9b`, `qwen3_6_27b`, `qwen3_6`, `qwen3_vl`, `internvl`,
`sensenova_qwen`, `cambrian_p`, `spatial_mllm`, `vlm3r`.
Use the same preset for setup and `OOS_MODEL`; environments are selected automatically.

VLM-3R builds its CUDA extension on the first GPU run. Load a CUDA 12.8 toolkit
module first if `nvcc` is not on `PATH`; `CUDA_HOME` is also recognized.

## Slurm

Adjust resources in [the job script](launchers/slurm_oos_eval.sh). After setup:

```bash
mkdir -p logs
OOS_MODEL=qwen3_5_9b OOS_LIMIT=2 OOS_OFFLINE=1 sbatch launchers/slurm_oos_eval.sh

# After setup.sh --all: test two questions per model.
OOS_OFFLINE=1 bash launchers/submit_oos_smoke_tests.sh --all --limit 2
```

Logs: `logs/oos_videoqa_<job-id>.{out,err}`. Omit `OOS_LIMIT` for a full single-model run.

## Configuration

No `.env` is required; [.env.example](.env.example) can save settings between sessions.
Model arguments: [launchers/models/](launchers/models/). Defaults: [launchers/oos_env.sh](launchers/oos_env.sh).
Setup uses manifest revisions when available and otherwise downloads `main`, printing the resolved commit.

## Acknowledgements

Built on lmms-eval. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).

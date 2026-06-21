# OOS Spatial Memory Evaluation

This repository is a customized LMMS-Eval-based evaluation repo for out-of-sight spatial memory in egocentric videos. It keeps the upstream LMMS-Eval pipeline, but adds our custom `oos_videoqa` task, video-prefix construction, multi-turn history modes, OOS-specific scoring, and model wrapper fixes for OOS-style video QA.

The original LMMS-Eval documentation is still useful for general framework internals, model registration, and task configuration. This README focuses only on the customized pieces used in this repo.

## New User Checklist

For a new machine or user, the usual setup path is:

1. Create and activate a Python environment, then install this repo.
2. Edit `lmms_eval/tasks/oos_videoqa/oos_videoqa_multi_turn.yaml` so `dataset_kwargs.data_files.test` points to your local OOS JSONL file.
3. Edit `launchers/oos_env.sh` for local paths and secrets: Hugging Face/Ollama tokens, cache directories, output directory, ffmpeg path, and `VLM3R_REPO` if using VLM-3R.
4. (Optional) Run a short smoke test with `OOS_LIMIT=2`.
5. Submit the full job with `launchers/slurm_oos_eval.sh` or run locally with `launchers/run_oos_eval.sh`.

## Environment Setup

Start from a fresh clone and create one Python environment inside the repo:

```bash
cd lmms-eval
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"
```

`.[all]` installs the main evaluation dependencies.

The OOS task uses `ffmpeg` to extract video prefixes and frames. First check it on the same machine where the evaluation will run:

```bash
ffmpeg -version
uname -m
```

If `ffmpeg -version` works, prefer that existing install:

```bash
export FFMPEG_PATH="$(command -v ffmpeg)"
```

If ffmpeg is not available, install or choose a binary that matches `uname -m`. For example, use an x86_64 build on x86_64 nodes and an aarch64/ARM build on aarch64 nodes. Do not copy a binary from a different architecture.

Install ffmpeg from the same architecture you will run on. If evaluation runs on a compute node, install or verify ffmpeg inside an interactive session on that node type, not only on the login node. 

(optional)If you are using a private/gated Hugging Face model or ollama cloud, set a token:

```bash
export HF_TOKEN=<your-token>
export OLLAMA_API_KEY=<your-api-key>
```

Run all commands below from inside `lmms-eval`.

## Model Setup

Most models use the base environment above. VLM-3R is the exception because it imports code from a separate `VLM-3R` repository.

| Model preset | `lmms-eval` model name | Extra setup |
| --- | --- | --- |
| `qwen3_6` | `qwen3_5` | Base env is enough. Uses Qwen-VL utilities and TorchCodec by default. |
| `qwen3_vl` | `qwen3_vl_chat_fixed` | Base env is enough. Uses Qwen-VL utilities and TorchCodec by default. |
| `llava` | `llava_onevision1_5_chat_fixed` | You may need to downgrade the transformer package version. |
| `internvl` | `internvl_hf_chat` | Base env plus any model-specific packages required by the selected InternVL checkpoint. |
| `phi4` | `phi4_multimodal_chat_fixed` | Base env plus any model-specific packages required by Phi-4 multimodal. |
| `vlm3r` | `vlm_3r` | Requires a local VLM-3R clone on `PYTHONPATH`. |

For VLM-3R, create an env and clone the external repo:

```bash
cd <your-workspace>
git clone https://github.com/VITA-Group/VLM-3R.git

cd lmms-eval
uv venv .venv-vlm3r
source .venv-vlm3r/bin/activate
uv pip install -e ".[all]"

# Install VLM-3R requirements according to that repository README.
# The local wrapper can decode videos with ffmpeg when Decord is unavailable.
# Common patterns are one of:
uv pip install -e ../VLM-3R
# or, if you want the full upstream environment:
uv pip install -r ../VLM-3R/requirements.txt

export VLM3R_REPO=<your-workspace>/VLM-3R
```

VLM-3R default checkpoints used by the launcher:

```bash
export VLM3R_CKPT=Journey9ni/vlm-3r-llava-qwen2-lora
export VLM3R_BASE=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2
```

## Custom Task

The main OOS task files are:

```text
lmms_eval/tasks/oos_videoqa/
|-- oos_videoqa_multi_turn.yaml
`-- utils.py
```

The task evaluates egocentric video question answering where the model must reason about objects, locations, and events after they move out of sight. Multi-turn examples are expanded into per-step samples by `utils.process_docs`.

The active task YAML is:

```text
lmms_eval/tasks/oos_videoqa/oos_videoqa_multi_turn.yaml
```

Important task hooks (don't change):

```yaml
task: oos_videoqa
process_docs: !function utils.process_docs
doc_to_visual: !function utils.oos_doc_to_visual
doc_to_text: !function utils.oos_doc_to_text
doc_to_messages: !function utils.oos_doc_to_messages_with_visuals
process_results: !function utils.oos_process_results
metric_list:
  - metric: oos_score
    aggregation: !function utils.oos_aggregate_results
cluster_key: source_video_id
```

## Data

The active task YAML is:

```text
lmms_eval/tasks/oos_videoqa/oos_videoqa_multi_turn.yaml
```

On a new machine, edit `dataset_kwargs.data_files.test` in that YAML so it points to your local OOS JSONL file, for example:

```yaml
dataset_kwargs:
  data_files:
    test: /work/courses/3dv/team1/data/vqa/selected_250_sorted_ranges.jsonl
```

## Visual Inputs

`utils.oos_doc_to_visual` builds the visual input for a sample. By default, it returns a video prefix ending at the query time. Depending on environment settings and question type, it can also return:

- no visual input for text-only/video-ablation runs
- a query-time frame image for step-1 debugging
- a marked prefix video for spatial-reference questions
- an additional BEV layout image for raw fixture-option debugging

Common visual settings:

```bash
export OOS_NO_VIDEO_INPUT=0
export OOS_VIDEO_CACHE_DIR=.cache/oos_video
export OOS_PREPROCESS_VIDEO=0
export OOS_TARGET_FPS=1
export OOS_RESIZE_WIDTH=448
export OOS_RESIZE_HEIGHT=448
export OOS_VIDEO_WIDTH=448
export OOS_VIDEO_HEIGHT=448
```

## History Modes

Set `OOS_HISTORY_MODE` before running:

```bash
export OOS_HISTORY_MODE=none   # no previous turns
export OOS_HISTORY_MODE=gold   # previous turns use ground-truth answers
export OOS_HISTORY_MODE=pred   # previous turns use earlier model predictions, if supported
```

`gold` and `none` are handled in the OOS task utilities. `pred` requires model-wrapper support. In this repo it is implemented for the OOS fixed Qwen, Phi, LLaVA, and VLM-3R paths. The current InternVL HF chat wrapper uses task-provided history and does not implement custom prediction-history injection.

Keep document shuffling disabled for `pred` mode so dependent steps are evaluated in order:

```bash
export LMMS_EVAL_SHUFFLE_DOCS=0
```

## Scoring

`utils.oos_process_results` scores both multiple-choice and open time/point predictions. `utils.oos_aggregate_results` reports overall accuracy and, when available, step-level and trajectory-level metrics.

Common scoring tolerances:

```bash
export OOS_TIME_TOLERANCE_SEC=3.0
export OOS_COORD_TOLERANCE_NORM=0.2
```

## Running

The launch path is organized under `launchers/`: shared logic lives in `common.sh`, model presets live in `models/`. The generic runner works on a laptop, workstation, or cluster node.

Configure the launcher environment in `launchers/oos_env.sh`. At minimum, check these values before running:

```bash
export HF_TOKEN=...                  # or provide SECRET_KEY/HF_TOKEN in your shell
export OLLAMA_API_KEY=...            # only needed for LiteLLM/Ollama-style runs
export HF_HOME=...
export HF_DATASETS_CACHE=...
export LMMS_EVAL_CACHE=...
export TMPDIR=...
export OOS_VIDEO_CACHE_DIR=...
export OOS_OUTPUT_DIR=...
export FFMPEG_PATH=...
export VLM3R_REPO=...                # only needed for OOS_MODEL=vlm3r
```

`launchers/run_oos_eval.sh` loads `launchers/oos_env.sh` automatically.

Run locally:

```bash
OOS_MODEL=qwen3_6 bash launchers/run_oos_eval.sh
OOS_MODEL=qwen3_vl bash launchers/run_oos_eval.sh
OOS_MODEL=llava bash launchers/run_oos_eval.sh
OOS_MODEL=internvl bash launchers/run_oos_eval.sh
OOS_MODEL=phi4 bash launchers/run_oos_eval.sh
OOS_MODEL=vlm3r VLM3R_REPO=<your-workspace>/VLM-3R bash launchers/run_oos_eval.sh
```

Run a short smoke test:

```bash
OOS_MODEL=qwen3_6 OOS_LIMIT=2 bash launchers/run_oos_eval.sh
```

Submit through Slurm:

```bash
OOS_MODEL=qwen3_6 OOS_VENV=.venv/bin/activate sbatch launchers/slurm_oos_eval.sh
```

`OOS_VENV` may be an absolute path or a path relative to the directory where you run `sbatch`. Edit the `#SBATCH` account, partition, GPU, and time lines in `launchers/slurm_oos_eval.sh` for your cluster. 

Outputs and logs are written under:

```text
lmms-eval/outputs/oos_videoqa/
lmms-eval/logs/
```

Override model arguments when needed:

```bash
OOS_MODEL=custom \
OOS_LMMS_MODEL=qwen3_5 \
OOS_MODEL_ARGS="pretrained=Qwen/Qwen3.6-27B,fps=1,max_num_frames=128,enable_thinking=False" \
bash launchers/run_oos_eval.sh
```

## Useful Debug Knobs

```bash
export OOS_DEBUG_STEP=5b                 # evaluate only selected step ids
export OOS_CHAT_DEBUG=1                  # print model-wrapper prompt/media diagnostics
export OOS_STEP1_QUERY_FRAME_DEBUG=0     # use a single query-time frame for step 1
export OOS_STEP4_RAW_FIXTURE_OPTIONS=0   # use raw fixture options and optional BEV image
```

If a run fails with an empty dataset error such as `Invalid key: 0 is out of bounds for size 0`, check whether `OOS_DEBUG_STEP` filtered out every expanded sample.

## Repository Notes

This repo is based on LMMS-Eval. The upstream project provides the core evaluator, task API, model registry, caching, logging, and many general model/task implementations. Our main changes are the OOS task utilities, OOS YAML, run scripts, and OOS-specific model wrapper behavior.

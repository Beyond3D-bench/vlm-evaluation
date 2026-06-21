# OOS Spatial Memory Evaluation

This repository is a customized LMMS-Eval-based evaluation repo for out-of-sight spatial memory in egocentric videos. It keeps the upstream LMMS-Eval pipeline, but adds our custom `oos_videoqa` task, video-prefix construction, multi-turn history modes, OOS-specific scoring, and model wrapper fixes for OOS-style video QA.

The original LMMS-Eval documentation is still useful for general framework internals, model registration, and task configuration. This README focuses only on the customized pieces used in this repo.

## Environment Setup

Start from a fresh clone and create one Python environment inside the repo:

```bash
cd lmms-eval
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"
```

`.[all]` installs the main evaluation dependencies plus the TorchCodec video backend. It does not install Decord. If you need Decord, use:

```bash
uv pip install -e ".[all,video-legacy]"
```

If you are using a private or gated Hugging Face model, set a token:

```bash
export HF_TOKEN=<your-token>
```

Run all commands below from inside `lmms-eval`.

## Model Setup

Most models use the base environment above. VLM-3R is the exception because it imports code from a separate `VLM-3R` repository.

| Model preset | `lmms-eval` model name | Extra setup |
| --- | --- | --- |
| `qwen3_6` | `qwen3_5` | Base env is enough. Uses `Qwen/Qwen3.6-27B` by default. |
| `qwen3_vl` | `qwen3_vl_chat_fixed` | Base env is enough. Uses Qwen-VL utilities and TorchCodec by default. |
| `llava` | `llava_onevision1_5_chat_fixed` | You may need to downgrade the transformer package version. |
| `internvl` | `internvl_hf_chat` | Base env plus any model-specific packages required by the selected InternVL checkpoint. |
| `phi4` | `phi4_multimodal_chat_fixed` | Base env plus any model-specific packages required by Phi-4 multimodal. |
| `vlm3r` | `vlm_3r` | Requires Decord and a local VLM-3R clone on `PYTHONPATH`. |

For VLM-3R, create the env with Decord support and clone the external repo:

```bash
cd <your-workspace>
git clone https://github.com/VITA-Group/VLM-3R.git

cd lmms-eval
uv venv .venv-vlm3r
source .venv-vlm3r/bin/activate
uv pip install -e ".[all,video-legacy]"

# Install VLM-3R requirements according to that repository README.
# Common patterns are one of:
uv pip install -e ../VLM-3R
# or:
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

Important task hooks:

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

The current YAML reads a local JSONL path:

```text
/work/courses/3dv/team1/data/vqa.jsonl
```

On a new machine, edit `lmms_eval/tasks/oos_videoqa/oos_videoqa_multi_turn.yaml` so `dataset_kwargs.data_files.test` points to your local JSONL file.

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
export OOS_RESIZE_WIDTH=224
export OOS_RESIZE_HEIGHT=224
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

## Message Function

The recommended default is:

```yaml
doc_to_messages: !function utils.oos_doc_to_messages_with_visuals
```

This builds the full chat prompt on the task side: system prompt, history turns, current question, and visual content. The fixed Qwen/LLaVA wrappers and the VLM-3R/InternVL chat paths can consume this cleaner message format. The wrappers still apply each model-specific chat template afterward.

## Scoring

`utils.oos_process_results` scores both multiple-choice and open time/point predictions. `utils.oos_aggregate_results` reports overall accuracy and, when available, step-level and trajectory-level metrics.

Common scoring tolerances:

```bash
export OOS_TIME_TOLERANCE_SEC=3.0
export OOS_COORD_TOLERANCE_NORM=0.2
```

## Running

The launch path is organized under `launchers/`: shared logic lives in `common.sh`, model presets live in `models/`, and ETH Team 1 shortcuts live in `team1/`. The generic runner works on a laptop, workstation, or cluster node.

Configure the environment:

```bash
# launchers/oos_env.sh is the committed Team 1 example config.
# Edit launchers/oos_env.sh for your HF token, cache paths, output path, and VLM3R_REPO if needed.
# launchers/run_oos_eval.sh loads this file automatically.
```

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
OOS_MODEL=qwen3_6 OOS_VENV=.venv-gb10/bin/activate sbatch launchers/slurm_oos_eval.sh
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

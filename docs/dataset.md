# Prepared evaluation datasets

The current loader reads local JSONL with one record per line. This reference
is for preparing data; see the [README](../README.md) to run evaluations.

Set `OOS_DATASET_JSONL` to your prepared file. See
[`examples/prepared_dataset.jsonl`](../examples/prepared_dataset.jsonl) for a
synthetic format example; its placeholder video path must be replaced to run it.

## Record format

| Field | Meaning |
| --- | --- |
| `doc_id` / `trajectory_id` | Record identifier; use unique identifiers. |
| `video_id` | Source video identifier, also used to infer kitchen layout images. |
| `video_path` | Path to the prepared video, accessible on the execution machine. |
| `query_time_sec` | Nonnegative query time in seconds in the supplied video. |
| `mode` | `multi_turn` for a trajectory containing `steps`; otherwise a single question. |
| `steps` | Ordered question objects for a trajectory. Skipped steps are omitted during evaluation. |
| `object_a_name` | Target object name used in prompt construction. |
| `target_reference_image_path` | Optional identity-reference image path. |
| `target_reference_steps` | Step identifiers that should receive the reference image. |

Each step has `step`, `step_question_class`, `question`, and an answer. For
multiple choice, supply string `choices` and a zero-based `correct_idx`, usually
with `target_text`. Open-answer questions use `target_text` or `answer`.
`depends_on_steps`, `branch_group`, `acceptable_idxs`, `acceptable_answers`,
`answer_metadata`, and `skipped` preserve the prepared task's dependencies and
scoring information. Dependencies and branch groups are retained for reporting;
they do not add previous answers to prompts. Keep the producer's metadata, particularly temporal and
coordinate references for open-answer steps 2/3 and fixture metadata for step 4.
Single-question records place question/answer fields at the top level.

## Media paths

Absolute paths must exist on the execution node. Relative paths resolve from
the repository root when using the launcher, not from the JSONL directory.
To relocate videos without rewriting JSONL, set `OOS_VIDEO_BASE_DIR`: the
loader joins that directory with each video's basename. Reference images use
their supplied paths directly.

Step-4 fixture questions can use kitchen layout images under
`$OOS_DATA_ROOT/HD-EPIC/kit_layout/<kitchen_id>.jpeg` (also `.jpg` or `.png`).
Set `OOS_KITCHEN_BEV_DIR` or `OOS_KITCHEN_BEV_IMAGE_PATH` to override this.

```bash
source launchers/oos_env.sh
python tools/validate_dataset.py "$OOS_DATASET_JSONL"
# Check only JSONL structure when media is mounted elsewhere:
python tools/validate_dataset.py "$OOS_DATASET_JSONL" --schema-only
```

The launcher checks structure and video/reference-image existence before model
loading. This lightweight check does not decode videos, verify frame timing, or
validate all task-specific scoring metadata or optional kitchen layout images.

## Protocol notes

`mode=multi_turn` describes storage: each unskipped step becomes an independent
question. Dependencies and trajectory IDs support grouped reporting; previous
answers are never added to prompts. `OOS_LIMIT` counts expanded questions.

Use the [README](../README.md#evaluation-settings) for evaluation defaults.
Direct `lmms_eval` invocation without the launcher can use different defaults.
Set `OOS_PREPROCESS_VIDEO=1` if media needs preprocessing; otherwise its frame
rate and dimensions should match the configured values.

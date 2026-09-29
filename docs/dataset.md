# Local dataset format

The standard BEYOND3D datasets are downloaded by `setup.sh`; no local dataset
file is needed for the normal evaluation. Use this guide when evaluating your own
questions with the same task and scoring code.

## Run a local dataset

Create a JSONL file with one JSON object per line. Put the referenced videos in one directory, then set:

```bash
export OOS_DATASET_JSONL=/absolute/path/to/questions.jsonl
export OOS_VIDEO_BASE_DIR=/absolute/path/to/videos
python tools/validate_dataset.py "$OOS_DATASET_JSONL"
OOS_MODEL=qwen3_5_9b OOS_OFFLINE=1 bash launchers/run_oos_eval.sh
```

The launcher validates the file again before evaluating it. For a schema check
before the videos are available, add `--schema-only` to the validation command.

## Single-question records

Each line must have these fields:

| Field | Meaning |
| --- | --- |
| `doc_id`, `trajectory_id`, or `id` | A nonempty record identifier. |
| `query_time_sec` | A finite, nonnegative number giving the query time in the video. |
| `video_path` | A nonempty video filename or path. When `OOS_VIDEO_BASE_DIR` is set, evaluation uses its basename in that directory. |
| `question` | Nonempty question text. |
| `choices` and `correct_idx`, or `target_text`/`answer` | For multiple choice, use a nonempty array of strings and its zero-based answer index. For an open question, give the expected answer text. |

For example, one line of `questions.jsonl` can be:

```json
{"doc_id":"example-1","query_time_sec":12.0,"video_path":"example.mp4","question":"Where is the cup?","choices":["On the table","In the sink"],"correct_idx":0}
```

`question_class` is optional. It groups results for macro accuracy; without it,
records use the `unknown` group. Prepare videos consistently with the
[benchmark protocol](../README.md#prepare-videos) if you want comparable results.

## Records with steps

The loader also accepts a record with `mode: "multi_turn"` and a nonempty
`steps` array. Each unskipped step needs a unique `step` identifier, a
`question`, and either `choices` with `correct_idx` or `target_text`/`answer`.
Shared `video_path` and `query_time_sec` stay on the outer record. The task
expands steps into independently evaluated questions; it does not pass earlier
answers into later prompts.

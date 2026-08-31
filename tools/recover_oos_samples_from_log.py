#!/usr/bin/env python3
"""Recover lmms-eval OOS sample JSONL records from an interrupted debug log.

The Slurm ``.err`` file only contains tqdm progress.  The paired ``.out`` file
contains one ``[STEP]``/``[CLEANED OUTPUT]`` block per completed request.  This
script joins those responses back to the source dataset named in the log and
uses the OOS task's normal expansion and scoring helpers to recreate the sample
JSONL schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful assistant trained to answer spatial and visual questions "
    "based on egocentric videos. The videos are captured from a first-person "
    "perspective and may contain various objects and interactions. Use the video "
    "to answer the question. The video is sampled at 1 frame per second."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        type=Path,
        help="Interrupted Slurm .err or paired .out log",
    )
    parser.add_argument("output", type=Path, help="Recovered sample JSONL path")
    return parser.parse_args()


def paired_stdout(path: Path) -> Path:
    if path.suffix == ".err":
        path = path.with_suffix(".out")
    if not path.is_file():
        raise FileNotFoundError(f"paired stdout log not found: {path}")
    return path


def parse_run(log_text: str) -> tuple[Path, list[tuple[int, str]]]:
    dataset_match = re.search(r"^Dataset JSONL:\s*(.+)$", log_text, re.MULTILINE)
    if not dataset_match:
        raise ValueError("could not find 'Dataset JSONL:' in stdout log")
    dataset_path = Path(dataset_match.group(1).strip())

    recovered: list[tuple[int, str]] = []
    for block in log_text.split("[BATCH]"):
        step_match = re.search(r"^\[STEP\].*?\bdoc_id=(\d+)\b", block, re.MULTILINE)
        response_match = re.search(
            r"^\[CLEANED OUTPUT\]\s*\n(.*?)(?=\n(?:\[|={20,})|\Z)",
            block,
            re.MULTILINE | re.DOTALL,
        )
        if step_match and response_match:
            recovered.append(
                (int(step_match.group(1)), response_match.group(1).strip())
            )

    ids = [doc_id for doc_id, _ in recovered]
    if len(ids) != len(set(ids)):
        raise ValueError("stdout log contains duplicate completed doc_id values")
    if not recovered:
        raise ValueError("no completed [STEP]/[CLEANED OUTPUT] blocks found")
    return dataset_path, recovered


def load_expanded_docs(dataset_path: Path):
    # Match the interrupted run recorded in this log. Import after setting the
    # environment because the task reads these switches at module import time.
    os.environ["OOS_HISTORY_MODE"] = "none"
    os.environ["OOS_DEBUG_STEP"] = "4,5a,5b,5c,5d"
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from lmms_eval.tasks.oos_videoqa import utils

    docs = []
    keep_steps = {"4", "5a", "5b", "5c", "5d"}
    with dataset_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            raw_doc = json.loads(line)
            if raw_doc.get("mode") == "multi_turn" and isinstance(
                raw_doc.get("steps"), list
            ):
                expanded = utils._expand_multi_turn_doc(raw_doc)
            else:
                expanded = [utils._normalize_single_turn_doc(raw_doc)]
            docs.extend(doc for doc in expanded if str(doc.get("step")) in keep_steps)
    return utils, docs


def build_sample(utils, doc_id: int, doc: dict, response: str) -> dict:
    score = utils.oos_process_results(doc, [response])
    input_text = utils.oos_doc_to_text(doc, {"system_prompt": SYSTEM_PROMPT})
    # lmms-eval hashes Instance.doc, which is None for this task/run.
    doc_hash = hashlib.sha256(
        json.dumps(None, indent=2, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    return {
        "doc_id": doc_id,
        "target": str(doc.get("answer")),
        "filtered_resps": response,
        # All recovered responses are one generated option letter. The Qwen
        # wrapper reports two output tokens for this form in completed runs.
        "token_counts": [{"output_tokens": 2}],
        "doc_hash": doc_hash,
        **score,
        "input": input_text,
    }


def main() -> int:
    args = parse_args()
    stdout_path = paired_stdout(args.log.resolve())
    log_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    dataset_path, recovered = parse_run(log_text)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"source dataset not found: {dataset_path}")

    utils, docs = load_expanded_docs(dataset_path)
    max_doc_id = max(doc_id for doc_id, _ in recovered)
    if max_doc_id >= len(docs):
        raise ValueError(
            f"log doc_id {max_doc_id} is outside expanded dataset of {len(docs)} rows"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for doc_id, response in sorted(recovered):
            sample = build_sample(utils, doc_id, docs[doc_id], response)
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(
        f"Recovered {len(recovered)} of {len(docs)} samples from {stdout_path} "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

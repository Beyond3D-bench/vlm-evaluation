#!/usr/bin/env python3
"""Check prepared OOS JSONL structure and referenced media without loading a model."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def validate_record(row, *, check_media=True):
    errors = []
    if not isinstance(row, dict):
        return ['record must be a JSON object']
    if not any(row.get(key) for key in ('doc_id', 'trajectory_id', 'id')):
        errors.append('missing doc_id, trajectory_id, or id')
    value = row.get('query_time_sec')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        errors.append('query_time_sec must be a finite nonnegative number')
    video = row.get('video_path')
    if not isinstance(video, str) or not video:
        errors.append('video_path must be a nonempty string')
    elif check_media:
        base = os.getenv('OOS_VIDEO_BASE_DIR')
        path = Path(base) / Path(video).name if base else Path(video)
        if not path.is_file():
            errors.append(f'video does not exist: {path}')
    reference = row.get('target_reference_image_path')
    if reference and check_media and not Path(reference).is_file():
        errors.append(f'target reference image does not exist: {reference}')
    if 'steps' in row and row.get('mode') != 'multi_turn':
        errors.append('records with steps must set mode=multi_turn')
    steps = row.get('steps') if 'steps' in row else [row]
    if not isinstance(steps, list) or not steps:
        return errors + ['steps must be a nonempty list when present']
    seen = set()
    for index, step in enumerate(steps):
        label = f'steps[{index}]' if 'steps' in row else 'question'
        if not isinstance(step, dict):
            errors.append(f'{label} must be an object')
            continue
        if step.get('skipped'):
            continue
        if 'steps' in row:
            step_id = str(step.get('step', '')).strip()
            if not step_id or step_id in seen:
                errors.append(f'{label}: missing or duplicate step identifier')
            seen.add(step_id)
        if not isinstance(step.get('question'), str) or not step['question'].strip():
            errors.append(f'{label}: missing question text')
        choices = step.get('choices')
        if choices:
            if not isinstance(choices, list) or not all(isinstance(c, str) for c in choices):
                errors.append(f'{label}: choices must be a list of strings')
            else:
                answer = step.get('correct_idx')
                if type(answer) is not int or not 0 <= answer < len(choices):
                    errors.append(f'{label}: correct_idx must index choices (zero-based)')
        elif step.get('target_text', step.get('answer')) in (None, ''):
            errors.append(f'{label}: missing target_text or answer')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--schema-only', action='store_true', help='Skip media existence checks.')
    args = parser.parse_args()
    count = failures = 0
    try:
        with args.dataset.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                count += 1
                try:
                    errors = validate_record(json.loads(line), check_media=not args.schema_only)
                except (ValueError, TypeError) as exc:
                    errors = [str(exc)]
                for error in errors:
                    print(f'{args.dataset}:{line_number}: {error}')
                failures += len(errors)
    except OSError as exc:
        parser.exit(1, f'{exc}\n')
    if not count:
        print('Dataset contains no records.')
        failures += 1
    print(f'Checked {count} records; {failures} errors.')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())

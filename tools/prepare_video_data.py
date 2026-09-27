#!/usr/bin/env python3
"""Select benchmark videos, optionally download originals, then preprocess them."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def collect_video_ids(jsonls: list[Path]) -> list[str]:
    ids = set()
    for path in jsonls:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                video_id = record.get('video_id', '')
                if not re.fullmatch(r'P0[1-9]-\d{8}-\d{6}', video_id):
                    raise ValueError(f'{path}:{line_number}: invalid video_id {video_id!r}')
                if record.get('video_path', f'{video_id}.mp4') != f'{video_id}.mp4':
                    raise ValueError(f'{path}:{line_number}: expected flat video_path {video_id}.mp4')
                ids.add(video_id)
    if not ids:
        raise ValueError('No video IDs found')
    return sorted(ids)


def video_is_ready(path: Path) -> bool:
    """Check readability and the format expected by evaluation."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (448, 448):
            return False
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if abs(cap.get(cv2.CAP_PROP_FPS) - 1) > 1e-6 or count < 1:
            return False
        if not cap.read()[0]:
            return False
        if count == 1:
            return True
        cap.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
        return cap.read()[0]
    finally:
        cap.release()


def check_calibrations(video_ids: list[str], root: Path) -> None:
    # This preflight runs before any original-video download.
    if __package__:
        from .preprocess_videos import calibrated_circle
    else:
        from preprocess_videos import calibrated_circle
    failures = []
    for video_id in video_ids:
        path = root / video_id.split('-', 1)[0] / video_id / 'device_calibration.json'
        try:
            calibrated_circle(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f'{video_id}: {exc}')
    if failures:
        raise SystemExit(
            f'Missing or invalid camera calibration for {len(failures)} videos in {root}.\n'
            + '\n'.join(failures[:5])
            + '\nPass --intermediate-archive with an HD-EPIC intermediate-data ZIP and rerun setup. '
              'See the Prepare videos section in README.md; set OOS_INTERMEDIATE_ROOT or --intermediate-root.'
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jsonl', type=Path, action='append', required=True, help='Repeat for both benchmark variants.')
    parser.add_argument('--data-root', type=Path, required=True, help='Parent of HD-EPIC/Videos.')
    parser.add_argument('--output-root', type=Path, required=True, help='Flat directory of processed MP4s.')
    mask = parser.add_mutually_exclusive_group(required=True)
    mask.add_argument('--intermediate-root', type=Path, help='Directory containing Pxx/video_id/device_calibration.json.')
    mask.add_argument('--fixed-circle', action='store_true', help='Legacy inscribed-circle mask; does not use calibration.')
    parser.add_argument('--intermediate-archive', action='append', help='ZIP URL or local path; extracted automatically.')
    parser.add_argument('--download', action='store_true', help='Download only the selected original MP4s, with MD5 verification.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--download-workers', type=int, default=8)
    parser.add_argument('--segments-per-file', type=int, default=4)
    parser.add_argument('--backend', choices=['ffmpeg', 'opencv'], default='ffmpeg')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--list-only', action='store_true', help='Write video_ids.txt without downloading or processing.')
    args = parser.parse_args()
    try:
        ids = collect_video_ids(args.jsonl)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if min(args.workers, args.download_workers, args.segments_per_file) < 1:
        parser.error('Worker and segment counts must be positive')
    args.output_root.mkdir(parents=True, exist_ok=True)
    id_file = args.output_root.parent / 'video_ids.txt'
    id_file.write_text('\n'.join(ids) + '\n')
    print(f'Selected {len(ids)} videos; IDs: {id_file}', flush=True)
    if args.list_only:
        return
    pending = ids if args.overwrite else [video_id for video_id in ids
                                            if not video_is_ready(args.output_root / f'{video_id}.mp4')]
    print(f'Reusing {len(ids) - len(pending)} valid videos; preparing {len(pending)}.', flush=True)
    if not pending:
        return
    if args.intermediate_archive:
        if not args.intermediate_root:
            parser.error('--intermediate-archive requires --intermediate-root')
        if __package__:
            from .download_hd_epic import install_intermediate_archive
        else:
            from download_hd_epic import install_intermediate_archive
        for archive in args.intermediate_archive:
            install_intermediate_archive(archive, args.intermediate_root)
    if args.intermediate_root:
        check_calibrations(pending, args.intermediate_root)
    pending_file = args.output_root.parent / 'video_ids_pending.txt'
    pending_file.write_text('\n'.join(pending) + '\n')
    tools = Path(__file__).resolve().parent
    if args.download:
        subprocess.run([sys.executable, str(tools / 'download_hd_epic.py'), '--participant', *sorted({video_id.split('-', 1)[0] for video_id in ids}),
                        '--output-path', str(args.data_root), '--workers', str(args.download_workers),
                        '--segments-per-file', str(args.segments_per_file)], check=True)
    command = [sys.executable, str(tools / 'preprocess_videos.py'), '--video-id-file', str(pending_file),
               '--video-root', str(args.data_root / 'HD-EPIC' / 'Videos'), '--output-root', str(args.output_root),
               '--target-fps', '1', '--target-size', '448', '448', '--watermark-style', 'token',
               '--backend', args.backend, '--workers', str(args.workers)]
    command += ['--intermediate-root', str(args.intermediate_root)] if args.intermediate_root else ['--fixed-circle']
    # The list contains only missing/invalid outputs (or explicit regeneration).
    command.append('--overwrite')
    subprocess.run(command, check=True)
    failed = [video_id for video_id in ids if not video_is_ready(args.output_root / f'{video_id}.mp4')]
    if failed:
        raise SystemExit('Prepared videos failed validation: ' + ', '.join(failed))
    print(f'All {len(ids)} benchmark videos are ready in {args.output_root}', flush=True)


if __name__ == '__main__':
    main()

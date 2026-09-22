import copy
import json
from pathlib import Path

from tools.validate_dataset import validate_record

EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/prepared_dataset.jsonl'


def record():
    return json.loads(EXAMPLE.read_text())


def test_synthetic_record_structure():
    assert validate_record(record(), check_media=False) == []


def test_invalid_answer_index_is_reported():
    row = record()
    row['steps'][0]['correct_idx'] = 2
    assert any('correct_idx' in error for error in validate_record(row, check_media=False))


def test_missing_media_is_reported():
    assert any('video does not exist' in error for error in validate_record(record()))


def test_relocated_video(tmp_path, monkeypatch):
    row = record()
    (tmp_path / 'synthetic_video.mp4').touch()
    monkeypatch.setenv('OOS_VIDEO_BASE_DIR', str(tmp_path))
    assert validate_record(row) == []


def test_duplicate_steps_are_reported():
    row = record()
    row['steps'].append(copy.deepcopy(row['steps'][0]))
    assert any('duplicate step' in error for error in validate_record(row, check_media=False))

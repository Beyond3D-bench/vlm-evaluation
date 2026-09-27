import copy

from tools.validate_dataset import validate_record


def record():
    return {
        "doc_id": "synthetic_001",
        "trajectory_id": "synthetic_001",
        "video_id": "synthetic_video",
        "mode": "multi_turn",
        "video_path": "/path/to/synthetic_video.mp4",
        "query_time_sec": 5.0,
        "object_a_name": "cup",
        "steps": [
            {
                "step": "1",
                "step_question_class": "oos_step1_visibility",
                "question": "Is the cup visible at the query time?",
                "choices": ["No", "Yes"],
                "correct_idx": 1,
                "target_text": "Yes",
                "skipped": False,
            }
        ],
    }


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

import pytest

from lmms_eval.tasks.oos_videoqa import utils


ALL_QUESTION_CLASSES = [
    "oos_step1_visibility",
    "oos_step2_last_visible",
    "oos_step3_last_placement",
    "oos_step4_fixture",
    "oos_branch_object_camera_relative_position",
    "oos_branch_object_camera_distance",
    "oos_branch_object_object_relation",
    "oos_branch_object_object_distance",
]


@pytest.mark.parametrize("question_class", ALL_QUESTION_CLASSES)
def test_last_frame_mode_uses_exact_query_time_for_every_question(
    monkeypatch, question_class
):
    monkeypatch.setattr(utils, "OOS_VIDEO_CONTEXT", "last_frame")
    monkeypatch.setenv("OOS_MARK_ANCHOR_OBJECT", "1")
    calls = []
    monkeypatch.setattr(
        utils,
        "_extract_query_frame_image",
        lambda path, timestamp, marker=None: calls.append((path, timestamp, marker))
        or "frame.jpg",
    )

    marker = [0.25, 0.75]
    doc = {
        "id": "sample",
        "step_question_class": question_class,
        "video_path": "video.mp4",
        "query_time_sec": 23,
        "answer_metadata": {"object_y_normalized_projected_pixel": marker},
    }

    assert utils.oos_doc_to_visual(doc) == ["frame.jpg"]
    expected_marker = (
        tuple(marker)
        if question_class
        in {
            "oos_branch_object_object_relation",
            "oos_branch_object_object_distance",
        }
        else None
    )
    assert calls == [("video.mp4", 23.0, expected_marker)]
    assert doc["video_context"] == "last_frame"


def test_query_frame_marker_is_drawn_on_the_single_image(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    monkeypatch.setenv("OOS_VIDEO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("OOS_MARKER_SIZE_PX", "8")
    monkeypatch.delenv("OOS_PREPROCESS_VIDEO", raising=False)
    commands = []
    monkeypatch.setattr(utils, "_run_ffmpeg", lambda cmd, _error: commands.append(cmd))

    output = utils._extract_query_frame_image(
        str(video_path), 23.0, marker_xy_norm=(0.25, 0.75)
    )

    assert output.endswith(".jpg")
    command = commands[0]
    assert command[command.index("-ss") + 1] == "23.0"
    assert command[command.index("-frames:v") + 1] == "1"
    vf = command[command.index("-vf") + 1]
    assert "drawbox=" in vf
    assert "color=red@0.95" in vf

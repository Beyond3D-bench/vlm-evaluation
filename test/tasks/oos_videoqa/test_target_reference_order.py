import pytest

from lmms_eval.tasks.oos_videoqa import utils


def _doc(reference_path):
    return {
        "id": "example",
        "step": 2,
        "target_reference_image_path": str(reference_path),
        "target_reference_steps": [2],
    }


def test_target_reference_defaults_to_after(tmp_path, monkeypatch):
    reference_path = tmp_path / "target.png"
    reference_path.touch()
    monkeypatch.delenv("OOS_TARGET_REFERENCE_POSITION", raising=False)

    visuals = utils._append_target_reference(_doc(reference_path), ["video.mp4"])

    assert visuals == ["video.mp4", str(reference_path)]


def test_target_reference_can_be_placed_before(tmp_path, monkeypatch):
    reference_path = tmp_path / "target.png"
    reference_path.touch()
    monkeypatch.setenv("OOS_TARGET_REFERENCE_POSITION", "before")

    visuals = utils._append_target_reference(_doc(reference_path), ["video.mp4"])

    assert visuals == [str(reference_path), "video.mp4"]


def test_target_reference_position_rejects_invalid_value(tmp_path, monkeypatch):
    reference_path = tmp_path / "target.png"
    reference_path.touch()
    monkeypatch.setenv("OOS_TARGET_REFERENCE_POSITION", "middle")

    with pytest.raises(ValueError, match="must be 'before' or 'after'"):
        utils._append_target_reference(_doc(reference_path), ["video.mp4"])

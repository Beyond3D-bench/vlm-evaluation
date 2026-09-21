from pathlib import Path

from tools.resolve_oos_dataset import resolve_dataset


def test_resolve_dataset_returns_snapshot_files(tmp_path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "videos").mkdir(parents=True)
    (snapshot / "vqa_baseline.jsonl").write_text("{}\n")
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        return snapshot

    jsonl, videos = resolve_dataset(
        repository="Ffffangzhu/BEYOND3D",
        revision="revision",
        cache_dir="/cache",
        offline=True,
        downloader=downloader,
    )

    assert jsonl == snapshot / "vqa_baseline.jsonl"
    assert videos == snapshot / "videos"
    assert calls == [{
        "repo_id": "Ffffangzhu/BEYOND3D",
        "repo_type": "dataset",
        "revision": "revision",
        "cache_dir": "/cache",
        "local_files_only": True,
    }]


def test_resolve_dataset_accepts_another_root_jsonl(tmp_path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "videos").mkdir(parents=True)
    (snapshot / "vqa_temporal_cues.jsonl").write_text("{}\n")

    jsonl, videos = resolve_dataset(
        repository="Ffffangzhu/BEYOND3D",
        revision=None,
        cache_dir=None,
        offline=True,
        jsonl_name="vqa_temporal_cues.jsonl",
        downloader=lambda **_: snapshot,
    )

    assert jsonl == snapshot / "vqa_temporal_cues.jsonl"
    assert videos == snapshot / "videos"

import json
from pathlib import Path

import pytest

from tools.prepare_video_data import collect_video_ids
from tools.preprocess_videos import calibrated_circle, preprocess_one
from tools.resolve_oos_dataset import resolve_dataset


def test_annotation_only_setup_does_not_require_videos(tmp_path):
    (tmp_path / 'vqa_baseline.jsonl').write_text('{}\n')
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return tmp_path
    resolve_dataset(repository='example/data', revision=None, cache_dir=None,
                    offline=False, downloader=download, annotations_only=True)
    assert calls[0]['ignore_patterns'] == ['videos/*']
    videos = tmp_path / 'local-videos'
    videos.mkdir()
    assert resolve_dataset(repository='example/data', revision=None, cache_dir=None,
                           offline=True, downloader=download,
                           video_base_dir=str(videos))[1] == videos


def test_video_selection_combines_variants_and_rejects_paths(tmp_path):
    first, second = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    a, b = 'P01-20240202-110250', 'P02-20240209-184316'
    first.write_text(json.dumps({'video_id': a}) + '\n')
    second.write_text('\n'.join(json.dumps({'video_id': v}) for v in (a, b)))
    assert collect_video_ids([first, second]) == [a, b]
    second.write_text(json.dumps({'video_id': a, 'video_path': '../outside.mp4'}))
    with pytest.raises(ValueError, match='flat video_path'):
        collect_video_ids([second])


@pytest.mark.parametrize('backend', ['ffmpeg', 'opencv'])
def test_calibrated_mask_and_video_encoding(tmp_path, monkeypatch, backend):
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    imageio = pytest.importorskip('imageio_ffmpeg')
    monkeypatch.setenv('FFMPEG_PATH', imageio.get_ffmpeg_exe())
    source, output = tmp_path / 'source.mp4', tmp_path / 'output.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (100, 100))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.full((100, 100, 3), 200, dtype=np.uint8))
    writer.release()
    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps({'cameras': {'camera-rgb': {
        'image_size': [100, 100], 'projection_params': [1, 30, 40], 'valid_radius': 20}}}))
    preprocess_one({'video_id': 'test', 'input_path': str(source), 'output_path': str(output),
                    'target_fps': 1, 'target_size': (448, 448), 'add_watermark': backend == 'opencv',
                    'watermark_style': 'plain', 'overwrite': False, 'backend': backend,
                    'calibration_geometry': calibrated_circle(calibration)})
    cap = cv2.VideoCapture(str(output))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(1)
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 2
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (448, 448, 3)
    assert frame[179, 134].mean() > 150  # Inside the off-center calibrated circle.
    assert frame[250, 300].max() < 15  # Fixed-circle geometry would retain this pixel.
    if backend == 'opencv':
        assert frame[415:, 320:].max() > 30  # Timestamp survives outside the masked region.


def test_reuses_valid_video_without_originals_or_calibration(tmp_path, monkeypatch):
    import sys
    from tools import prepare_video_data as prepare
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    vid = 'P01-20240202-110250'
    output = tmp_path / 'videos'
    output.mkdir()
    writer = cv2.VideoWriter(str(output / f'{vid}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 1, (448, 448))
    writer.write(np.zeros((448, 448, 3), dtype=np.uint8))
    writer.release()
    annotation = tmp_path / 'vqa.jsonl'
    annotation.write_text(json.dumps({'video_id': vid}))
    monkeypatch.setattr(sys, 'argv', ['prepare', '--jsonl', str(annotation), '--data-root', str(tmp_path / 'missing-originals'),
                                   '--output-root', str(output), '--intermediate-root', str(tmp_path / 'missing-calibration'), '--download'])
    monkeypatch.setattr(prepare.subprocess, 'run', lambda *a, **k: pytest.fail('Existing valid videos must not trigger download/preprocessing'))
    prepare.main()


def test_missing_calibration_stops_before_downloading(tmp_path, monkeypatch):
    import sys
    from tools import prepare_video_data as prepare
    annotation = tmp_path / 'vqa.jsonl'
    annotation.write_text(json.dumps({'video_id': 'P01-20240202-110250'}))
    monkeypatch.setattr(sys, 'argv', ['prepare', '--jsonl', str(annotation), '--data-root', str(tmp_path / 'data'),
                                   '--output-root', str(tmp_path / 'videos'), '--intermediate-root', str(tmp_path / 'calibration'), '--download'])
    monkeypatch.setattr(prepare.subprocess, 'run', lambda *a, **k: pytest.fail('Must preflight calibration before downloading'))
    with pytest.raises(SystemExit, match='Missing or invalid camera calibration'):
        prepare.main()


def test_dataset_stage_uses_both_variants_and_configured_paths(tmp_path, monkeypatch):
    import sys
    from tools import setup_dataset as setup
    for name in ('vqa_baseline.jsonl', 'vqa_temporal_cues.jsonl'):
        (tmp_path / name).write_text('{}')
    calls = []
    monkeypatch.setattr(setup, 'resolve_dataset', lambda **kwargs: (tmp_path / 'vqa_baseline.jsonl', tmp_path))
    monkeypatch.setenv('OOS_DATA_ROOT', str(tmp_path / 'data'))
    monkeypatch.setenv('OOS_VIDEO_BASE_DIR', str(tmp_path / 'local-videos'))
    monkeypatch.setenv('OOS_VIDEO_PREP_WORKERS', '2')
    monkeypatch.setattr(sys, 'argv', ['setup', '--intermediate-root', str(tmp_path / 'camera')])
    monkeypatch.setattr(setup.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    setup.main()
    command = calls[0]
    assert command.count('--jsonl') == 2
    assert str(tmp_path / 'vqa_temporal_cues.jsonl') in command
    assert command[command.index('--output-root') + 1] == str(tmp_path / 'local-videos')
    assert command[command.index('--workers') + 1] == '2'
    assert command[command.index('--backend') + 1] == 'opencv'


def test_participant_downloads_select_only_mp4s():
    from tools.download_hd_epic import FileEntry, choose_entries
    paths = ['Videos/P01/a.mp4', 'Videos/P02/b.mp4', 'Videos/P01/readme.txt',
             'Digital-Twin/P01/mesh.zip', 'SLAM-and-Gaze/P01/data.zip', 'readme.txt']
    selected = choose_entries([FileEntry(path, 'checksum') for path in paths], ['P01', 'P02'])
    assert [entry.rel_path for entry in selected] == paths[:2]


def test_intermediate_archive_extracts_without_requiring_a_layout(tmp_path):
    import zipfile
    from tools.download_hd_epic import install_intermediate_archive
    archive = tmp_path / 'P01.zip'
    video = 'P01-20240202-110250'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr(f'Downloaded folder/{video}/device_calibration.json', '{}')
        bundle.writestr(f'Downloaded folder/{video}/framewise_info.jsonl', '{}\n')
    root = tmp_path / 'automatic-location'
    install_intermediate_archive(str(archive), root)
    assert (root / 'P01' / video / 'device_calibration.json').read_text() == '{}'
    assert (root / 'P01' / video / 'framewise_info.jsonl').is_file()


def test_intermediate_archive_rejects_sign_in_page(tmp_path, monkeypatch):
    import io
    from tools import download_hd_epic as download
    response = io.BytesIO(b'<html>Sign in</html>')
    response.headers = {'Content-Type': 'text/html'}
    monkeypatch.setattr(download.urllib.request, 'urlopen', lambda *a, **k: response)
    with pytest.raises(ValueError, match='web/sign-in page'):
        download.install_intermediate_archive('https://example.org/folder', tmp_path / 'data')


def test_intermediate_archive_rejects_path_traversal(tmp_path):
    import zipfile
    from tools.download_hd_epic import install_intermediate_archive
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('../outside', 'data')
    with pytest.raises(ValueError, match='Unsafe archive member'):
        install_intermediate_archive(str(archive), tmp_path / 'data')
    assert not (tmp_path / 'outside').exists()


def test_intermediate_archive_downloads_zip_url(tmp_path, monkeypatch):
    import io
    import zipfile
    from tools import download_hd_epic as download
    content = io.BytesIO()
    with zipfile.ZipFile(content, 'w') as bundle:
        bundle.writestr('Intermediate_data/P02/P02-20240209-184316/device_calibration.json', '{}')
    response = io.BytesIO(content.getvalue())
    response.headers = {'Content-Type': 'application/zip'}
    monkeypatch.setattr(download.urllib.request, 'urlopen', lambda *a, **k: response)
    output = tmp_path / 'data'
    download.install_intermediate_archive('https://example.org/data.zip', output)
    assert (output / 'P02/P02-20240209-184316/device_calibration.json').is_file()


def test_preparation_downloads_by_participant(tmp_path, monkeypatch):
    import sys
    from tools import prepare_video_data as prepare
    annotation = tmp_path / 'vqa.jsonl'
    annotation.write_text(json.dumps({'video_id': 'P01-20240202-110250'}))
    monkeypatch.setattr(sys, 'argv', ['prepare', '--jsonl', str(annotation), '--data-root', str(tmp_path / 'data'),
                                   '--output-root', str(tmp_path / 'videos'), '--intermediate-root', str(tmp_path / 'calibration'), '--download'])
    readiness = iter([False, True])
    monkeypatch.setattr(prepare, 'video_is_ready', lambda _: next(readiness))
    monkeypatch.setattr(prepare, 'check_calibrations', lambda *a: None)
    commands = []
    monkeypatch.setattr(prepare.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    assert prepare.main() is None
    assert commands[0][commands[0].index('--participant') + 1] == 'P01'
    assert '--video-ids-file' not in commands[0]

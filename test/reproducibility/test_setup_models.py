import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.build_vlm3r_curope import find_cuda
from tools.prepare_cut3r import prepare_checkpoint
from tools.setup_models import selected_profiles

ROOT = Path(__file__).resolve().parents[2]


def test_all_models_share_only_four_environments():
    manifest = json.loads((ROOT / 'environments/profiles.json').read_text())
    profiles = selected_profiles(list(manifest['preset_profiles']), manifest)
    assert set(profiles) == {'base', 'cambrian-p', 'spatial-mllm', 'vlm3r'}
    assert len(profiles) == 4
    with pytest.raises(ValueError, match='Unknown models'):
        selected_profiles(['typo'], manifest)


def test_setup_dry_run_does_not_create_storage(tmp_path):
    storage = tmp_path / 'new-storage'
    env = dict(os.environ, OOS_STORAGE_ROOT=str(storage), OOS_VENV_ROOT=str(storage / 'venvs'))
    output = subprocess.check_output([sys.executable, str(ROOT / 'tools/setup_models.py'), '--all', '--dry-run'], env=env, text=True)
    assert not storage.exists()
    assert output.count('--profile base') == 1
    assert '--model vlm3r' in output
    assert 'resolve_oos_dataset.py' in output
    assert 'prepare_cut3r.py' in output


def test_cuda_detected_from_nvcc_symlink(tmp_path):
    root = tmp_path / 'cuda'
    (root / 'bin').mkdir(parents=True)
    nvcc = root / 'bin/nvcc'
    nvcc.touch()
    link = tmp_path / 'nvcc'
    link.symlink_to(nvcc)
    assert find_cuda({}, which=lambda _: str(link)) == root


def test_explicit_cuda_takes_precedence_and_invalid_path_fails(tmp_path):
    root = tmp_path / 'cuda'
    (root / 'bin').mkdir(parents=True)
    (root / 'bin/nvcc').touch()
    assert find_cuda({'CUDA_HOME': str(root)}, which=lambda _: None) == root
    with pytest.raises(RuntimeError, match='no bin/nvcc'):
        find_cuda({'VLM3R_CUDA_HOME': str(tmp_path / 'missing'), 'CUDA_HOME': str(root)})


def test_failed_download_is_not_reused(tmp_path):
    target = tmp_path / 'weights.pth'

    def fail(partial):
        partial.write_bytes(b'incomplete')
        raise RuntimeError('connection lost')

    with pytest.raises(RuntimeError, match='connection lost'):
        prepare_checkpoint(target, fail)
    assert not target.exists()
    assert not target.with_name(target.name + '.part').exists()

    def success(partial):
        partial.write_bytes(b'0' * (1024 * 1024))

    assert prepare_checkpoint(target, success) == target
    # A prepared checkpoint must not trigger a second download.
    assert prepare_checkpoint(target, fail) == target


def test_html_download_is_rejected(tmp_path):
    target = tmp_path / 'weights.pth'
    with pytest.raises(RuntimeError, match='HTML'):
        prepare_checkpoint(target, lambda partial: partial.write_bytes(b'<html>' + b' ' * (1024 * 1024)))
    assert not target.exists()


def test_downloader_allows_main_but_strict_mode_rejects_it():
    command = [sys.executable, str(ROOT / 'tools/download_hf_models.py'),
               '--model', 'qwen3_6_35b_a3b', '--dry-run']
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'Qwen/Qwen3.6-35B-A3B@main' in result.stdout
    strict = subprocess.run(command + ['--require-pinned'], cwd=ROOT, text=True, capture_output=True)
    assert strict.returncode != 0
    assert 'do not have pinned revisions' in strict.stderr


def test_build_detects_gpu_architecture_and_uses_active_python(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import build_vlm3r_curope as build

    cuda = tmp_path / 'cuda'
    (cuda / 'bin').mkdir(parents=True)
    (cuda / 'bin/nvcc').touch()
    source = tmp_path / 'repo/CUT3R/src/croco/models/curope'
    source.mkdir(parents=True)
    (source / 'setup.py').touch()
    monkeypatch.setenv('VLM3R_REPO', str(tmp_path / 'repo'))
    monkeypatch.setenv('CUDA_HOME', str(cuda))
    monkeypatch.delenv('VLM3R_CUDA_HOME', raising=False)
    monkeypatch.delenv('TORCH_CUDA_ARCH_LIST', raising=False)
    monkeypatch.setitem(sys.modules, 'curope', None)
    gpu = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                          get_device_capability=lambda _: (8, 0))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=gpu))
    monkeypatch.setattr(build.subprocess, 'check_output', lambda *a, **kw: 'Cuda compilation tools, release 12.8, V12.8.0')
    calls = []
    monkeypatch.setattr(build.subprocess, 'run', lambda *a, **kw: calls.append((a, kw)))
    assert build.main() == 0
    assert calls[0][0][0][0] == sys.executable
    assert calls[0][1]['env']['TORCH_CUDA_ARCH_LIST'] == '8.0'
    assert calls[0][1]['env']['CUDA_HOME'] == str(cuda)
    assert calls[0][1]['cwd'] == source


def test_build_rejects_wrong_cuda_version(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import build_vlm3r_curope as build

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'curope', None)
    monkeypatch.setattr(build, 'find_cuda', lambda _: tmp_path)
    monkeypatch.setattr(build.subprocess, 'check_output', lambda *a, **kw: 'release 12.4, V12.4.0')
    with pytest.raises(RuntimeError, match='CUDA 12.8 is required'):
        build.main()

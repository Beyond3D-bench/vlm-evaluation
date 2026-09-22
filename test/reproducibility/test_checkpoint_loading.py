import json
import os
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

from tools.resolve_checkpoint import resolve_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def test_local_checkpoint_never_downloads(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    download = Mock()
    assert resolve_checkpoint(str(tmp_path), downloader=download) == str(tmp_path)
    download.assert_not_called()


def test_cache_root_is_not_a_checkpoint(tmp_path):
    (tmp_path / 'snapshots').mkdir()
    with pytest.raises(ValueError, match='snapshots/<revision>'):
        resolve_checkpoint(str(tmp_path))


@pytest.mark.parametrize('offline', [False, True])
def test_hub_resolution_uses_manifest_pin_and_requested_network_mode(offline):
    download = Mock(return_value='/cache/snapshot')
    assert resolve_checkpoint('Qwen/Qwen3.5-9B', offline=offline, downloader=download) == '/cache/snapshot'
    assert download.call_args.kwargs['revision'] == 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
    assert download.call_args.kwargs['local_files_only'] is offline


def test_missing_absolute_directory_does_not_become_hub_id(tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        resolve_checkpoint(str(tmp_path / 'missing'))


@pytest.mark.parametrize('offline', ['0', '1'])
@pytest.mark.parametrize('preset', ['qwen3_vl', 'qwen3_6_35b_a3b', 'qwen3_5_9b', 'qwen3_6_27b', 'internvl', 'cambrian_p', 'spatial_mllm', 'sensenova_qwen'])
def test_presets_respect_loading_mode_without_forcing_staging(tmp_path, offline, preset):
    env = {'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'OOS_MODEL': preset, 'OOS_OFFLINE': offline, 'REPO_DIR': str(ROOT), 'LAUNCHER_DIR': str(ROOT / 'launchers')}
    script = '''set -eu
source "$LAUNCHER_DIR/oos_env.sh"
source "$LAUNCHER_DIR/common.sh"
load_oos_model_preset
printf '%s\\n' "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "$OOS_STAGE_MODEL_TO_TMP" "$MODEL_ARGS"
'''
    output = subprocess.check_output(['bash', '-c', script], cwd=tmp_path, env=env, text=True).splitlines()
    assert output[:3] == [offline, offline, '0']
    if preset != 'spatial_mllm':
        assert f'local_files_only={"True" if offline == "1" else "False"}' in output[3]
    assert '/scratch/' not in output[3].split(',')[0]


def test_generic_local_override_reaches_model_and_staging(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    env = dict(os.environ, REPO_DIR=str(ROOT), LAUNCHER_DIR=str(ROOT / 'launchers'), OOS_PRETRAINED=str(tmp_path))
    script = '''set -eu
source "$LAUNCHER_DIR/common.sh"
MODEL_PRESET=qwen3_vl
MODEL_ARGS=pretrained=Qwen/Qwen3-VL-8B-Instruct,fps=1
resolve_oos_checkpoint
printf '%s\\n' "$MODEL_ARGS" "$OOS_STAGE_MODEL_DIR"
'''
    output = subprocess.check_output(['bash', '-c', script], env=env, text=True).splitlines()
    assert output == [f'pretrained={tmp_path},fps=1', str(tmp_path)]


def test_vlm3r_local_overrides_survive_preset_loading(tmp_path):
    checkpoint = tmp_path / 'vlm-3r-llava-qwen2-lora'
    base = tmp_path / 'LLaVA-NeXT-Video-7B-Qwen2'
    cudnn = tmp_path / 'cudnn'
    cudnn.mkdir()
    for directory in (checkpoint, base):
        directory.mkdir()
        (directory / 'config.json').write_text('{}')
    env = {
        'PATH': os.environ['PATH'], 'HOME': str(tmp_path),
        'REPO_DIR': str(ROOT), 'LAUNCHER_DIR': str(ROOT / 'launchers'),
        'OOS_MODEL': 'vlm3r', 'OOS_OFFLINE': '1',
        'VLM3R_CUDNN_LIB': str(cudnn),
        'VLM3R_SIGLIP_SOURCE': str(base),
        'OOS_MODEL_ARGS': f'pretrained={checkpoint},model_base={base},for_get_frames_num=8',
    }
    script = '''set -eu
source "$LAUNCHER_DIR/oos_env.sh"
source "$LAUNCHER_DIR/common.sh"
load_oos_model_preset
printf '%s\\n' "$MODEL_ARGS" "$TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
'''
    output = subprocess.check_output(['bash', '-c', script], cwd=tmp_path, env=env, text=True).splitlines()
    model_args, weights_only_compat = output[-2:]
    assert f'pretrained={checkpoint}' in model_args
    assert f'model_base={base}' in model_args
    assert 'for_get_frames_num=8' in model_args
    assert weights_only_compat == '1'


def test_offline_missing_snapshot_does_not_retry_online():
    download = Mock(side_effect=FileNotFoundError('not cached'))
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint('Qwen/Qwen3.5-9B', offline=True, downloader=download)
    assert download.call_count == 1
    assert download.call_args.kwargs['local_files_only'] is True

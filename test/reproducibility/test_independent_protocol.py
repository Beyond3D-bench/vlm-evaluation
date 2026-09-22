"""Protocol checks without importing model runtimes or downloading datasets."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / 'lmms_eval/tasks/oos_videoqa/utils.py'


def load_task(monkeypatch):
    # These tests exercise pure record/prompt/scoring functions. Stub only their
    # unused dataset/dataframe/logging dependencies to keep CPU CI lightweight.
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(Dataset=list))
    monkeypatch.setitem(sys.modules, 'pandas', SimpleNamespace(DataFrame=object))
    monkeypatch.setitem(sys.modules, 'loguru', SimpleNamespace(logger=Mock()))
    spec = importlib.util.spec_from_file_location('oos_protocol_under_test', TASK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def task(monkeypatch):
    monkeypatch.delenv('OOS_HISTORY_MODE', raising=False)
    return load_task(monkeypatch)


def question():
    return {
        'id': 'trajectory__step_2', 'trajectory_id': 'trajectory',
        'step': '2', 'step_question_class': 'oos_step2_last_visible',
        'question': 'When was the cup last visible?',
        'choices': ['00:00:01', '00:00:02'], 'answer_idx': 0,
        'history_messages': [{'role': 'assistant', 'content': [{'type': 'text', 'text': 'PRIOR_ANSWER_SENTINEL'}]}],
        'gold_history_messages': [{'role': 'assistant', 'content': 'GOLD_SENTINEL'}],
    }


def test_prompts_ignore_supplied_history_and_current_answer(task):
    doc = question()
    options = {'system_prompt': 'Answer the visual question.'}
    messages = task.oos_doc_to_messages(doc, options)
    assert [m['role'] for m in messages] == ['system', 'user']
    assert messages[1]['content'][0]['text'] == (
        'Question: When was the cup last visible?\nOptions:\n'
        'A. 00:00:01\nB. 00:00:02\nSelect the best option and output only its letter.'
    )
    assert 'SENTINEL' not in task.oos_doc_to_text(doc, options)
    doc['answer_idx'] = 1
    assert task.oos_doc_to_messages(doc, options) == messages


def test_expansion_preserves_reporting_metadata_without_history(task):
    row = {
        'trajectory_id': 'trajectory', 'doc_id': 'trajectory',
        'video_id': 'video', 'video_path': '/video.mp4', 'query_time_sec': 2,
        'mode': 'multi_turn', 'include_gold_history': True,
        'gold_history_messages': [{'role': 'assistant', 'content': 'GOLD_SENTINEL'}],
        'steps': [
            {'step': '1', 'question': 'Earlier?', 'target_text': 'SECRET_EARLIER_ANSWER'},
            {'step': '2', 'question': 'Now?', 'choices': ['Yes', 'No'], 'correct_idx': 0,
             'branch_group': 'branch'},
        ],
    }
    docs = task._expand_multi_turn_doc(row)
    assert len(docs) == 2
    assert docs[1]['trajectory_id'] == 'trajectory'
    assert docs[1]['branch_group'] == 'branch'
    prompt = task.oos_doc_to_text(docs[1], {'system_prompt': 'System'})
    assert 'SECRET_EARLIER_ANSWER' not in prompt
    assert 'GOLD_SENTINEL' not in prompt
    assert task.oos_process_results(docs[1], ['A'])['oos_score']['accuracy'] == 1.0
    assert task.oos_process_results(docs[1], ['B'])['oos_score']['accuracy'] == 0.0


@pytest.mark.parametrize('mode', ['gold', 'pred', 'typo'])
def test_direct_task_loading_rejects_unsupported_modes(monkeypatch, mode):
    monkeypatch.setenv('OOS_HISTORY_MODE', mode)
    with pytest.raises(ValueError, match='independent questions'):
        load_task(monkeypatch)


@pytest.mark.parametrize('mode', ['none', 'gold', 'pred'])
def test_launcher_rejects_unsupported_modes_before_model_loading(tmp_path, mode):
    env_file = tmp_path / 'env.sh'
    env_file.write_text(f'export OOS_HISTORY_MODE={mode}\n')
    env = dict(os.environ, OOS_ENV_FILE=str(env_file), LAUNCHER_DIR=str(ROOT / 'launchers'))
    result = subprocess.run(['bash', '-c', 'set -e; source "$LAUNCHER_DIR/common.sh"; load_oos_env'], env=env, text=True, capture_output=True)
    assert (result.returncode == 0) == (mode == 'none')
    if mode != 'none':
        assert 'independent-question' in result.stderr

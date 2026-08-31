from types import SimpleNamespace

import torch
from PIL import Image
from transformers.cache_utils import DynamicCache

from lmms_eval.models.chat.stream3d_vlm import Stream3DVLM


class _TemplateProcessor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "PREFIX<|im_end|>\n"


def _bare_stream3d(*, timestamps=False):
    model = object.__new__(Stream3DVLM)
    model.stream_frame_timestamps = timestamps
    model.processor = _TemplateProcessor()
    return model


def test_prefix_dynamic_cache_can_be_restored_after_a_question_branch():
    cache = DynamicCache()
    prefix_key = torch.randn(1, 2, 5, 4)
    prefix_value = torch.randn(1, 2, 5, 4)
    cache.update(prefix_key, prefix_value, layer_idx=0)
    cache.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4), layer_idx=0)

    cache.crop(5)

    assert cache.get_seq_length() == 5
    if hasattr(cache, "layers"):
        restored_key = cache.layers[0].keys
        restored_value = cache.layers[0].values
    else:
        restored_key, restored_value = cache[0]
    assert torch.equal(restored_key, prefix_key)
    assert torch.equal(restored_value, prefix_value)


def test_clear_prefix_cache_does_not_discard_decoded_frames():
    model = _bare_stream3d()
    frames = [object()]
    model._stream_media_cache_key = ("video",)
    model._stream_media_cache_frames = frames
    model._stream_prefix_cache_key = ("prefix",)
    model._stream_prefix_cache_past = object()
    model._stream_prefix_cache_seq_len = 123

    model._clear_stream_prefix_cache()

    assert model._stream_prefix_cache_key is None
    assert model._stream_prefix_cache_past is None
    assert model._stream_prefix_cache_seq_len == 0
    assert model._stream_media_cache_frames is frames


def test_prefix_fingerprint_reuses_only_identical_pre_query_frames():
    model = _bare_stream3d()
    model._stream_prefix_fingerprint_key = None
    model._stream_prefix_fingerprint_value = None
    black = Image.new("RGB", (2, 2), "black")
    white = Image.new("RGB", (2, 2), "white")
    red = Image.new("RGB", (2, 2), "red")

    base = model._prefix_frame_fingerprint([black, white], 1, ("base",))
    different_query_frame = model._prefix_frame_fingerprint([black.copy(), red], 1, ("marked",))
    different_prefix = model._prefix_frame_fingerprint([white, red], 1, ("other",))

    assert base == different_query_frame
    assert base != different_prefix


def test_terminal_query_frame_uses_stop_delimiter():
    model = _bare_stream3d()
    model.stream_frame_policy = "auto"
    model.frame_end_token_id = 198
    model.frame_interval_token_id = 11

    assert model._chosen_decision_token(sampled_token_id=785, has_more_frames=False) == 198
    assert model._chosen_decision_token(sampled_token_id=785, has_more_frames=True) == 785


def test_first_prefix_chunk_keeps_official_image_comma_order():
    model = _bare_stream3d()
    frames = [object(), object()]

    text = model._prefix_chunk_text(frames, chunk_start=0, system="system")

    assert text == "PREFIX"
    content = model.processor.messages[1]["content"]
    assert [item["type"] for item in content] == ["image", "text", "image", "text"]
    assert content[0]["image"] is frames[0]
    assert content[1]["text"] == ","
    assert content[2]["image"] is frames[1]
    assert content[3]["text"] == ","


def test_continuation_prefix_chunk_keeps_absolute_frame_timestamps():
    model = _bare_stream3d(timestamps=True)

    text = model._prefix_chunk_text([object(), object()], chunk_start=4, system="unused")

    assert text == (
        "<TIME 00:00:04.0 video 1> <|vision_start|><|image_pad|><|vision_end|>,"
        "<TIME 00:00:05.0 video 1> <|vision_start|><|image_pad|><|vision_end|>,"
    )


def test_inference_optimizations_keep_only_last_logits_and_skip_attention_weights():
    model = _bare_stream3d()
    lm_head = torch.nn.Linear(4, 3, bias=False)
    attention = torch.nn.MultiheadAttention(4, 2, batch_first=True)
    block = SimpleNamespace(cross_attention=attention)
    model._model = SimpleNamespace(
        lm_head=lm_head,
        feature_fusion=SimpleNamespace(cross_attn_blocks=[block]),
    )

    model._install_inference_optimizations()

    logits = lm_head(torch.randn(1, 5, 4))
    output, weights = attention(
        torch.randn(1, 2, 4),
        torch.randn(1, 3, 4),
        torch.randn(1, 3, 4),
    )
    assert logits.shape == (1, 1, 3)
    assert output.shape == (1, 2, 4)
    assert weights is None

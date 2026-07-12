# Chat Model Backends

This directory contains the chat-template model backends for `lmms_eval`.
These wrappers are used when a task provides `doc_to_messages` and expects a
conversation-style prompt with typed multimodal content.

The common input contract is:

```python
[
    {
        "role": "system" | "user" | "assistant",
        "content": [
            {"type": "text", "text": "..."},
            {"type": "image", "url": image_or_path},
            {"type": "video", "url": video_path},
            {"type": "audio", "url": audio_path},
        ],
    },
]
```

`lmms_eval.protocol.ChatMessages` validates this structure and converts it to
backend-specific formats:

- `to_hf_messages()` for Hugging Face processors and model chat templates.
- `to_openai_messages()` for OpenAI-compatible APIs, encoding images and video
  frames as `image_url` payloads.
- `to_qwen3_vl_openai_messages()` for Qwen3-VL-style APIs that need timestamped
  video frames.
- `extract_media()` for backends that handle media separately from text.

Most backends implement `generate_until()` and return either strings or
`GenerationResult` objects with optional `TokenCounts`. `loglikelihood()` and
`generate_until_multi_round()` are generally not implemented here unless the
backend explicitly supports them.

## Registration

Chat backends are exposed through `AVAILABLE_CHAT_TEMPLATE_MODELS` in
`lmms_eval/models/__init__.py`. The model id passed to `--model` is usually the
key in that mapping, not necessarily the internal `@register_model(...)` name.

| `--model` id | File | Main class | Purpose |
| --- | --- | --- | --- |
| `async_hf_model` | `async_hf_model.py` | `AsyncHFModel` | Multi-worker local Hugging Face chat generation without accelerate/torchrun. |
| `async_openai` | `async_openai.py` | `AsyncOpenAIChat` | Async OpenAI-compatible chat API client with retries, optional adaptive concurrency, and video-frame conversion. |
| `bagel_lmms_engine` | `bagel_lmms_engine.py` | `BagelLmmsEngine` | BAGEL via `lmms-engine`, including image generation output saved to disk. |
| `cambrian_p` | `cambrian_p.py` | `CambrianP` | Cambrian-P chat wrapper for pose-grounded video QA, loaded through the Cambrian package. |
| `huggingface` | `huggingface.py` | `Huggingface` | Generic Hugging Face chat-template wrapper for `AutoProcessor` plus `AutoModel*`. |
| `internvl_hf` | `internvl_hf.py` | `InternVLHf` | InternVL Hugging Face wrapper with explicit frame sampling and patch controls. |
| `litellm_chat` | `litellm_chat.py` | `LiteLLMChatCompatible` | LiteLLM-backed OpenAI-compatible chat wrapper for many hosted/local providers. |
| `llava_hf` | `llava_hf.py` | `LlavaHf` | Chat wrapper around the simple LLaVA HF backend. |
| `llava_onevision1_5` | `llava_onevision1_5.py` | `Llava_OneVision1_5` | Chat wrapper around the simple LLaVA-OneVision 1.5 backend. |
| `llava_onevision1_5_chat_fixed` | `llava_onevision1_5_chat_fixed.py` | `Llava_OneVision1_5_Chat_Fixed` | LLaVA-OneVision 1.5 chat wrapper with OOS/multi-turn media-history fixes. |
| `longvila` | `longvila.py` | `LongVila` | VLLM-based LongVILA chat wrapper. |
| `nanovlm` | `nanovlm.py` | `NanoVLM` | Local NanoVLM wrapper with internal multi-GPU worker dispatch. |
| `openai` | `openai.py` | `OpenAICompatible` | Synchronous OpenAI-compatible chat API wrapper. |
| `phi4_multimodal` | `phi4_multimodal.py` | `Phi4` | Chat wrapper around the simple Phi-4 multimodal backend. |
| `phi4_multimodal_chat_fixed` | `phi4_multimodal_chat_fixed.py` | `Phi4ChatFixed` | Phi-4 multimodal chat wrapper with OOS/multi-turn media-history fixes. |
| `qwen2_5_vl` | `qwen2_5_vl.py` | `Qwen2_5_VL` | Qwen2.5-VL chat wrapper using `qwen_vl_utils`. |
| `qwen3_vl` | `qwen3_vl.py` | `Qwen3_VL` | Qwen3-VL chat wrapper using the simple Qwen3-VL model defaults. |
| `qwen3_vl_chat_fixed` | `qwen3_vl_chat_fixed.py` | `Qwen3_VL_Chat_Fixed` | Qwen3-VL chat wrapper with OOS/multi-turn media-history fixes. |
| `qwen3_5` | `qwen3_5.py` | `Qwen3_5` | Qwen3.5 variant that combines Qwen3.5 defaults with the fixed Qwen3-VL chat path. |
| `sglang` | `sglang.py` | `Sglang` | SGLang runtime backend. The class is registered internally as `sglang_runtime`. |
| `stream3d_vlm` | `stream3d_vlm.py` | `Stream3DVLM` | Stream3D-VLM wrapper with OOS history and stream-frame prompt controls. |
| `thyme` | `thyme.py` | `Thyme` | Qwen2.5-VL-based reasoning wrapper with iterative code execution for image tasks. |
| `vllm` | `vllm.py` | `VLLM` | VLLM `chat()` backend that sends OpenAI-style multimodal messages. |
| `vllm_generate` | `vllm_generate.py` | `VLLMGenerate` | VLLM `generate()` backend that builds processed multimodal inputs directly. |
| `vlm_3r` | `vlm_3r.py` | `VLM3R` | VLM-3R/LLaVA-NeXT-Video wrapper with optional point-cloud export. |

## Typical Usage

Run a chat backend the same way as other `lmms_eval` models, but choose a task
whose YAML resolves to `doc_to_messages`.

```bash
python -m lmms_eval \
  --model huggingface \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct,batch_size=1,max_num_frames=32 \
  --tasks <task_name> \
  --batch_size 1
```

For an OpenAI-compatible endpoint:

```bash
OPENAI_API_KEY=... OPENAI_API_BASE=https://example/v1 \
python -m lmms_eval \
  --model async_openai \
  --model_args model=gpt-4o-mini,adaptive_concurrency=true,max_retries=5,nframes=32 \
  --tasks <task_name> \
  --batch_size 1
```

For a local VLLM chat backend:

```bash
python -m lmms_eval \
  --model vllm \
  --model_args model=Qwen/Qwen2.5-VL-3B-Instruct,tensor_parallel_size=1,batch_size=1,nframes=32 \
  --tasks <task_name> \
  --batch_size 1
```

For Cambrian-P:

```bash
python -m lmms_eval \
  --model cambrian_p \
  --model_args pretrained=nyu-visionx/Cambrian-P-8B,conv_template=qwen_2,video_max_frames=32 \
  --tasks <task_name> \
  --batch_size 1
```

## Common Model Args

Not every backend accepts every argument, but these names are used repeatedly:

- `pretrained` or `model`: model identifier or local checkpoint path.
- `device`, `device_map`: local placement for Hugging Face-style backends.
- `batch_size`: per-backend batch size. Several wrappers force or require
  `batch_size=1` because they handle video or multi-turn state per sample.
- `attn_implementation`: usually one of `flash_attention_2`, `sdpa`, or `eager`
  for Hugging Face models that support it.
- `max_num_frames`, `nframes`, `max_frames`, `max_frame_num`, `fps`: video
  sampling controls. The exact accepted names differ by backend.
- `max_pixels`, `min_pixels`, `min_image_pixels`: image/video frame resolution
  controls for Qwen/VLLM/OpenAI-style conversion paths.
- `max_new_tokens`, `temperature`, `top_p`, `num_beams`: generation controls.
- `system_prompt` or `reasoning_prompt`: optional prompt injection for wrappers
  that support a global system or reasoning prompt.

Unknown `model_args` are handled differently by backend: some ignore and warn,
some assert or raise. Check the target class constructor before adding new args.

## Backend Notes

### Hugging Face-family wrappers

`huggingface.py`, `async_hf_model.py`, `internvl_hf.py`, `nanovlm.py`, and the
Qwen/LLaVA/Phi chat variants load local models and processors. They usually:

1. Call the task's `doc_to_messages`.
2. Validate with `ChatMessages`.
3. Apply the model processor's chat template.
4. Extract media and pass it to the processor/model-specific generation path.

`async_hf_model` and `nanovlm` manage their own worker devices. Run them without
accelerate/torchrun multi-process launch. Use `worker_gpus=0,1` or
`worker_count=N` to select internal workers.

### Cambrian-P

`cambrian_p.py` consumes chat `doc_to_messages` and converts typed image/video
content into Cambrian image-token prompts plus Cambrian visual tensors. It uses
the same Cambrian package loading and video processing utilities as the existing
Cambrian-S simple backend, but preserves chat history instead of relying on
`doc_to_visual`.

Common args:

- `pretrained`: Cambrian-P checkpoint path or Hugging Face id.
- `conv_template`: Cambrian conversation template, default `qwen_2`.
- `video_max_frames`, `video_fps`, `video_force_sample`: video sampling controls.
- `miv_token_len`, `si_token_len`, `image_aspect_ratio`, `anyres_max_subimages`:
  Cambrian visual-token/image-shape controls.

Cambrian-P currently returns text generation only; pose-head outputs are not
surfaced through the `lmms_eval` generation API.

### OpenAI-compatible wrappers

`openai.py`, `async_openai.py`, and `litellm_chat.py` convert chat messages into
OpenAI-style payloads. Images are base64-encoded; videos are decoded into frame
images through `qwen_vl_utils`.

Useful environment variables:

- `OPENAI_API_KEY`, `OPENAI_API_BASE`: default credentials and endpoint.
- `LMMS_IMAGE_ENCODE_FORMAT`: image encoding format for API payloads
  (`PNG` by default; `JPEG`, `JPG`, and `WEBP` enable quality settings).
- `LMMS_IMAGE_JPEG_QUALITY`: quality for lossy image encodings.
- `OOS_CHAT_DEBUG=1`: enables additional debug output in selected OOS-focused
  wrappers.

`async_openai` supports `adaptive_concurrency=true`, prefix-aware dispatch, and
Qwen3-VL timestamp formatting through `is_qwen3_vl=true`.

### VLLM and SGLang wrappers

`vllm.py` uses VLLM's chat API and sends OpenAI-style messages. `vllm_generate.py`
uses VLLM's generate API and constructs multimodal input dictionaries directly,
which is useful when the chat path does not process a model's video metadata
correctly. `sglang.py` starts an SGLang `Engine` and supports optional MCP tool
configuration through `mcp_server_path`.

### OOS/multi-turn fixed wrappers

`qwen3_vl_chat_fixed.py`, `phi4_multimodal_chat_fixed.py`, and
`llava_onevision1_5_chat_fixed.py` contain extra logic for OOS VideoQA style
multi-turn evaluation. They distinguish between media already embedded by
`doc_to_messages` and media loaded from `doc_to_visual`, preserve or rebuild
history depending on the question class, and optionally track predicted history.

`stream3d_vlm.py` also has OOS-specific controls:

- `oos_history_mode`: `none`, `gold`, or `pred`.
- `stream_prompt_mode`: `query_after_prefix` or `query_before_video`.
- `stream_frame_policy`: `auto`, `all`, or timestamp-driven options.

### Specialized wrappers

- `bagel_lmms_engine.py` requires `lmms-engine` and can save generated images
  under `output_image_dir`.
- `thyme.py` uses iterative reasoning and sandboxed code execution for image
  tasks. It currently expects a single user image and warns if `batch_size` is
  not `1`.
- `vlm_3r.py` loads VLM-3R/LLaVA-NeXT-Video components, samples video with
  decord or ffmpeg, and can export point clouds when enabled.

## Adding a New Chat Backend

1. Add a module in this directory.
2. Subclass `lmms_eval.api.model.lmms`.
3. Set `is_simple = False`.
4. Register the class with `@register_model("<internal_name>")`.
5. Implement `generate_until(self, requests)` using each request's arguments:
   `(ctx, doc_to_messages, gen_kwargs, doc_id, task, split)`.
6. Convert `doc_to_messages(doc)` through `ChatMessages` unless the backend has
   a strong reason to handle the raw structure itself.
7. Return `GenerationResult(text=..., token_counts=...)` when token usage is
   known.
8. Add the backend to `AVAILABLE_CHAT_TEMPLATE_MODELS` in
   `lmms_eval/models/__init__.py` so `--model <id>` can resolve it.

Keep model-specific prompt repair or dataset-specific behavior local to the
smallest relevant wrapper. Prefer generic `ChatMessages` conversion for common
HF/OpenAI/VLLM formatting.

## Dependencies

Optional dependencies are imported lazily in many wrappers. Common optional
packages include:

- `transformers`, `accelerate`, `torch` for local Hugging Face models.
- `qwen-vl-utils` and `decord` for video frame extraction.
- `openai` and `litellm` for API-backed models.
- `vllm` for VLLM backends.
- `sglang` for the SGLang runtime backend.
- `lmms-engine` for BAGEL.
- `cambrian` from the Cambrian/Cambrian-S package for Cambrian-P.
- Model-specific repositories for Stream3D-VLM and VLM-3R.

If an import is optional, the wrapper usually logs a warning and fails only when
the missing feature is used.

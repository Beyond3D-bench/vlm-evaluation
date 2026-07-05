import gc
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.load_video import read_video
from lmms_eval.protocol import ChatMessages


@register_model("stream3d_vlm_chat")
class Stream3DVLM(lmms):
    """Chat wrapper for Stream3D-VLM using the official qwen_vl inference path."""

    is_simple = False

    def _debug_enabled(self) -> bool:
        return os.getenv("OOS_CHAT_DEBUG", "0") == "1"

    def _dbg(self, msg: str) -> None:
        if self._debug_enabled() and self.rank == 0:
            print(msg, flush=True)

    def _content_preview(self, content: List[Dict[str, Any]], max_text_chars: int = 2000) -> str:
        parts = []
        for item in content:
            ctype = item.get("type")
            if ctype == "text":
                text = str(item.get("text", "")).replace("\n", " ").strip()
                if len(text) > max_text_chars:
                    text = text[:max_text_chars] + "..."
                parts.append(f"text='{text}'")
            elif ctype == "video":
                parts.append(f"video='{item.get('url', item.get('video', ''))}'")
            elif ctype == "image":
                image = item.get("image", item.get("url", ""))
                if isinstance(image, Image.Image):
                    parts.append(f"image=PIL(size={image.size}, mode={image.mode})")
                else:
                    parts.append(f"image='{image}'")
            else:
                parts.append(str(item))
        return " | ".join(parts)

    def _tensor_shape_summary(self, inputs: Dict[str, Any]) -> str:
        parts = []
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                parts.append(f"{key}={tuple(value.shape)}:{value.dtype}:{value.device}")
            elif isinstance(value, list):
                shapes = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        shapes.append(f"{tuple(item.shape)}:{item.dtype}:{item.device}")
                    else:
                        shapes.append(type(item).__name__)
                parts.append(f"{key}=[{', '.join(shapes)}]")
            else:
                parts.append(f"{key}={type(value).__name__}")
        return " ".join(parts)

    def _memory_summary(self) -> str:
        rss_mb = None
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        rss_mb = int(line.split()[1]) / 1024
                        break
        except OSError:
            pass

        parts = []
        if rss_mb is not None:
            parts.append(f"rss={rss_mb:.1f}MB")
        if torch.cuda.is_available():
            device = self.device if self.device.type == "cuda" else torch.device("cuda:0")
            try:
                parts.append(f"cuda_alloc={torch.cuda.memory_allocated(device) / 1024**3:.2f}GB")
                parts.append(f"cuda_reserved={torch.cuda.memory_reserved(device) / 1024**3:.2f}GB")
                parts.append(f"cuda_peak={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}GB")
            except Exception as exc:
                parts.append(f"cuda_mem_error={type(exc).__name__}:{exc}")
        return " ".join(parts) or "memory=unavailable"

    def _cleanup_memory(self, label: str) -> None:
        before = self._memory_summary()
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as exc:
                self._dbg(f"[MEM CLEANUP ERROR] {label}: {type(exc).__name__}: {exc}")
        after = self._memory_summary()
        eval_logger.info(f"Stream3D memory cleanup {label}: before={before} after={after}")

    def __init__(
        self,
        pretrained: str = "JonnyYu828/Stream3D-VLM-4B",
        device: str = "cuda",
        device_map: Optional[str] = None,
        batch_size: int = 1,
        max_frames: int = 32,
        video_decoder: str = "auto",
        torch_dtype: str = "bfloat16",
        attn_implementation: Optional[str] = "sdpa",
        trust_remote_code: bool = True,
        require_geometry: bool = True,
        stream3d_repo: Optional[str] = None,
        oos_history_mode: Optional[str] = None,
        stream_prompt_mode: Optional[str] = None,
        stream_frame_policy: Optional[str] = None,
        stream_frame_timestamps: Optional[Any] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        if int(batch_size) != 1:
            raise ValueError("Stream3DVLM currently supports batch_size=1.")

        self.path = pretrained
        self.batch_size_per_gpu = 1
        self.max_frames = int(max_frames)
        self.video_decoder = video_decoder.strip().lower()
        self.default_max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.stream3d_repo = stream3d_repo or os.getenv("STREAM3D_VLM_REPO", "/work/courses/3dv/team1/Stream3D-VLM")
        stream3d_src = Path(self.stream3d_repo) / "src"
        if stream3d_src.is_dir() and str(stream3d_src) not in sys.path:
            sys.path.insert(0, str(stream3d_src))

        self.oos_history_mode = (oos_history_mode or os.getenv("OOS_HISTORY_MODE", "gold")).strip().lower()
        if self.oos_history_mode not in {"none", "gold", "pred"}:
            raise ValueError("oos_history_mode must be one of: none, gold, pred")
        self._oos_pred_history: Dict[Tuple[str, str], str] = {}

        self.stream_prompt_mode = (stream_prompt_mode or os.getenv("STREAM3D_PROMPT_MODE", "query_after_prefix")).strip().lower()
        prompt_mode_aliases = {
            "after": "query_after_prefix",
            "query_after": "query_after_prefix",
            "query_after_prefix": "query_after_prefix",
            "before": "query_before_video",
            "query_before": "query_before_video",
            "question_before": "query_before_video",
            "query_before_video": "query_before_video",
            "question_before_video": "query_before_video",
        }
        self.stream_prompt_mode = prompt_mode_aliases.get(self.stream_prompt_mode, self.stream_prompt_mode)
        if self.stream_prompt_mode not in {"query_after_prefix", "query_before_video"}:
            raise ValueError("stream_prompt_mode must be one of: query_after_prefix, query_before_video")

        self.stream_frame_policy = (stream_frame_policy or os.getenv("STREAM3D_FRAME_POLICY", "auto")).strip().lower()
        frame_policy_aliases = {
            "native": "auto",
            "model": "auto",
            "auto": "auto",
            "all": "all",
            "force_all": "all",
            "read_all": "all",
        }
        self.stream_frame_policy = frame_policy_aliases.get(self.stream_frame_policy, self.stream_frame_policy)
        if self.stream_frame_policy not in {"auto", "all"}:
            raise ValueError("stream_frame_policy must be one of: auto, all")

        frame_timestamps_value = stream_frame_timestamps
        if frame_timestamps_value is None:
            frame_timestamps_value = os.getenv("STREAM3D_FRAME_TIMESTAMPS", "0")
        if isinstance(frame_timestamps_value, str):
            self.stream_frame_timestamps = frame_timestamps_value.strip().lower() in {"1", "true", "yes", "on", "time", "times", "timestamp", "timestamps"}
        else:
            self.stream_frame_timestamps = bool(frame_timestamps_value)

        accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(weeks=52))])
        self.accelerator = accelerator
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes
        self._device = torch.device(f"cuda:{self._rank}" if accelerator.num_processes > 1 else device)

        dtype = getattr(torch, torch_dtype, torch.bfloat16)
        model_kwargs = {"torch_dtype": dtype, "trust_remote_code": trust_remote_code, **kwargs}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if accelerator.num_processes > 1:
            model_kwargs["device_map"] = {"": self._rank}
        elif device_map:
            model_kwargs["device_map"] = device_map

        model_cls = self._model_class(require_geometry=require_geometry)
        eval_logger.info(f"Loading Stream3D-VLM with {model_cls.__name__} from {pretrained}")
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code, padding_side="left")
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code, padding_side="left")
        self._model = model_cls.from_pretrained(pretrained, **model_kwargs).eval()
        if "device_map" not in model_kwargs:
            self._model = self._model.to(self.device)
        self._config = self.model.config
        self._load_stream3d_preprocess()

        self.frame_interval_token_id = self.tokenizer.encode(",", add_special_tokens=False)[0]
        self.frame_end_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.im_start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.assistant_token_ids = self.tokenizer.encode("assistant", add_special_tokens=False)
        self.newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]

    @staticmethod
    def _model_class(require_geometry: bool = True):
        try:
            from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT

            return Qwen2_5_VLForConditionalGenerationWithVGGT
        except ImportError as exc:
            if require_geometry:
                raise ImportError(
                    "Stream3D-VLM requires qwen_vl.model.modeling_qwen2_5_vl."
                    "Qwen2_5_VLForConditionalGenerationWithVGGT. Set STREAM3D_VLM_REPO "
                    "to the official Stream3D-VLM checkout so its src directory can be imported."
                ) from exc

        import transformers

        for name in (
            "AutoModelForMultimodalLM",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ):
            model_cls = getattr(transformers, name, None)
            if model_cls is not None:
                eval_logger.warning(
                    "Falling back to %s; Stream3D geometry weights will not be used.",
                    name,
                )
                return model_cls

        raise ImportError(
            "This Transformers install cannot load Stream3D-VLM with either the "
            "geometry-aware class or a multimodal fallback."
        )

    def _load_stream3d_preprocess(self) -> None:
        try:
            from qwen_vl.data.utils import load_and_preprocess_images
        except ImportError as exc:
            raise ImportError(
                "Could not import qwen_vl.data.utils.load_and_preprocess_images "
                f"from Stream3D-VLM repo at {self.stream3d_repo}."
            ) from exc
        self._load_and_preprocess_images = load_and_preprocess_images

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return getattr(self.tokenizer, "eos_token_id", None)

    @property
    def pad_token_id(self):
        return getattr(self.tokenizer, "pad_token_id", self.eot_token_id)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _decode_video(self, path: str) -> List[Image.Image]:
        backend = self.video_decoder
        if backend == "auto":
            backend = os.getenv("FORCE_QWENVL_VIDEO_READER") or os.getenv("LMMS_VIDEO_DECODE_BACKEND") or "torchcodec"
        start = time.time()
        frames = read_video(path, num_frm=self.max_frames, backend=backend, force_include_last_frame=True)
        pil_frames = [Image.fromarray(frame).convert("RGB") for frame in frames]
        self._dbg(
            f"[VIDEO DECODE] path={path} backend={backend} requested_max_frames={self.max_frames} "
            f"decoded_frames={len(pil_frames)} elapsed={time.time() - start:.3f}s"
        )
        return pil_frames

    def _media_to_frames(self, messages: ChatMessages) -> List[Image.Image]:
        images, videos, _ = messages.extract_media()
        if videos and images:
            raise ValueError(
                "Stream3D prefix-streaming mode expects one video only; "
                "auxiliary images such as BEV must be disabled or handled separately."
            )
        frames: List[Image.Image] = []
        for video in videos:
            frames.extend(self._decode_video(str(video)))
        for image in images:
            frames.append(image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB"))
        if not frames:
            raise ValueError("Stream3D-VLM received no image/video input")
        kept = frames[: self.max_frames]
        sample_sizes = [frame.size for frame in kept[:5]]
        self._dbg(
            f"[MEDIA TO FRAMES] extracted_images={len(images)} extracted_videos={len(videos)} "
            f"total_frames={len(frames)} kept_frames={len(kept)} sample_sizes={sample_sizes}"
        )
        return kept

    def _prompt_text(self, messages: ChatMessages) -> str:
        parts: List[str] = []
        for msg in messages.messages:
            if msg.role == "system":
                continue
            text = "\n".join(c.text for c in msg.content if getattr(c, "type", None) == "text" and c.text.strip()).strip()
            if text:
                parts.append(f"Assistant: {text}" if msg.role == "assistant" else text)
        return "\n\n".join(parts)

    def _system_text(self, messages: ChatMessages) -> str:
        parts: List[str] = []
        for msg in messages.messages:
            if msg.role == "system":
                text = "\n".join(c.text for c in msg.content if getattr(c, "type", None) == "text" and c.text.strip()).strip()
                if text:
                    parts.append(text)
        return "\n\n".join(parts) or "You are a helpful assistant."

    def _should_use_geometry(self) -> bool:
        return bool(getattr(self.model.config, "use_geometry_encoder", False))

    def _frames_processor_inputs(self, text: str, frames: List[Image.Image]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        image_inputs = []
        geometry_encoder_inputs = []
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size

        for frame in frames:
            image_tensor = self._load_and_preprocess_images([frame])[0]
            geometry_encoder_inputs.append(image_tensor.clone())

            _, height, width = image_tensor.shape
            if (width // patch_size) % merge_size > 0:
                width = width - (width // patch_size) % merge_size * patch_size
            if (height // patch_size) % merge_size > 0:
                height = height - (height // patch_size) % merge_size * patch_size
            image_inputs.append(image_tensor[:, :height, :width])

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        return inputs, torch.stack(geometry_encoder_inputs).to(self.device)

    def _frame_processor_inputs(self, text: str, frame: Image.Image) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self._frames_processor_inputs(text, [frame])

    def _oos_num_images_before_query(self, doc: Optional[Dict[str, Any]], num_frames: int) -> int:
        if num_frames <= 1:
            return 0
        if doc and str(doc.get("video_context", "")).strip().lower() == "full":
            query_time = doc.get("stream3d_query_time_sec", doc.get("query_time_in_clip_sec", doc.get("query_time_sec")))
            try:
                query_frame_idx = int(round(float(query_time)))
            except (TypeError, ValueError):
                query_frame_idx = num_frames - 1
            return min(max(query_frame_idx, 0), num_frames - 1)
        return num_frames - 1

    def _sample_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature > 0:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _generate(self, messages: ChatMessages, gen_kwargs: Dict[str, Any], doc: Optional[Dict[str, Any]] = None) -> Tuple[str, int, float]:
        self._dbg("[FINAL PROTOCOL MESSAGES]")
        for msg_idx, msg in enumerate(messages.model_dump()["messages"]):
            content = msg.get("content", [])
            self._dbg(
                f"  [PROTO] idx={msg_idx} role={msg.get('role')} "
                f"types={[c.get('type') for c in content]} {self._content_preview(content)}"
            )
        frames = self._media_to_frames(messages)
        prompt = self._prompt_text(messages)
        system = self._system_text(messages)
        self._dbg(f"[SYSTEM] {system}")
        self._dbg(f"[PROMPT] {prompt}")
        self._dbg(f"[GEN KWARGS RAW] {gen_kwargs}")
        start = time.time()
        num_images_before_query = self._oos_num_images_before_query(doc, len(frames))
        self._dbg(
            f"[STREAM3D QUERY SPLIT] num_images_before_query={num_images_before_query} "
            f"num_images_after_query={len(frames) - num_images_before_query} "
            f"query_time_in_clip_sec={doc.get('query_time_in_clip_sec') if doc else None} "
            f"query_time_sec={doc.get('query_time_sec') if doc else None}"
        )
        text, tokens = self._generate_streaming(frames, prompt, system, gen_kwargs, num_images_before_query)
        elapsed = time.time() - start
        self._dbg(
            f"[GENERATION METRICS] elapsed={elapsed:.4f}s output_tokens={tokens} "
            f"speed={tokens / elapsed if elapsed > 0 else 0.0:.2f} tok/s"
        )
        self._dbg("\n" + "=" * 80)
        self._dbg("[RAW OUTPUT]")
        self._dbg(text)
        return text, tokens, elapsed

    def _generate_streaming(
        self,
        frames: List[Image.Image],
        prompt: str,
        system: str,
        gen_kwargs: Dict[str, Any],
        num_images_before_query: int = 0,
    ) -> Tuple[str, int]:
        max_new_tokens = int(gen_kwargs.get("max_new_tokens") or self.default_max_new_tokens)
        model_to_use = self.model.module if hasattr(self.model, "module") else self.model
        num_images_before_query = min(max(int(num_images_before_query), 0), len(frames) - 1)
        if self.stream_prompt_mode == "query_before_video":
            num_images_before_query = 0
            images_before = []
            images_after = frames
        else:
            images_before = frames[:num_images_before_query]
            images_after = frames[num_images_before_query:]
        first_image_after = images_after[0]
        remaining_images_after = images_after[1:]
        self._dbg(
            f"[STREAM3D GENERATION CONFIG] frames={len(frames)} max_new_tokens={max_new_tokens} "
            f"images_before={len(images_before)} images_after={len(images_after)} "
            f"prompt_mode={self.stream_prompt_mode} frame_policy={self.stream_frame_policy} "
            f"temperature={self.temperature} use_geometry={self._should_use_geometry()} "
            f"model_class={type(model_to_use).__name__}"
        )

        past_key_values = None
        current_seq_len = 0

        def _forward_inputs(inputs: Dict[str, torch.Tensor], geometry_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
            nonlocal past_key_values, current_seq_len
            token_len = inputs["input_ids"].shape[1]
            attention_mask = inputs["attention_mask"]
            cache_position = None
            if past_key_values is not None:
                past_len = past_key_values[0][0].shape[-2]
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=self.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=1)
                cache_position = torch.arange(current_seq_len, current_seq_len + token_len, device=self.device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                outputs = model_to_use(
                    input_ids=inputs["input_ids"],
                    attention_mask=attention_mask,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    geometry_encoder_inputs=[geometry_inputs] if self._should_use_geometry() and geometry_inputs is not None else None,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            current_seq_len += token_len
            return logits

        def _forward_tokens(token_ids: List[int]) -> None:
            nonlocal past_key_values, current_seq_len
            if not token_ids:
                return
            token_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                outputs = model_to_use(
                    input_ids=token_tensor,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    cache_position=torch.arange(current_seq_len, current_seq_len + token_tensor.shape[1], device=self.device),
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
            current_seq_len += token_tensor.shape[1]

        def _chosen_decision_token(sampled_token_id: int, has_more_frames: bool) -> int:
            if self.stream_frame_policy == "all":
                return self.frame_interval_token_id if has_more_frames else self.frame_end_token_id
            return sampled_token_id

        def _frame_time_token(frame_idx: int) -> str:
            seconds = float(frame_idx)
            hh = int(seconds // 3600)
            mm = int((seconds % 3600) // 60)
            ss = seconds % 60
            return f"<TIME {hh:02d}:{mm:02d}:{ss:04.1f} video 1>"

        def _image_text(frame_idx: int) -> str:
            prefix = f"{_frame_time_token(frame_idx)} " if self.stream_frame_timestamps else ""
            return f"{prefix}<|vision_start|><|image_pad|><|vision_end|>"

        def _prompt_first_image_text(frame_idx: int) -> str:
            if self.stream_frame_timestamps:
                return f"{prompt}\n{_frame_time_token(frame_idx)}"
            return f"{prompt},"

        if images_before:
            first_content = []
            if self.stream_frame_timestamps:
                first_content.append({"type": "text", "text": _frame_time_token(0)})
            first_content.append({"type": "image", "image": images_before[0]})
            first_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": first_content},
            ]
            text = self.processor.apply_chat_template(first_messages, tokenize=False, add_generation_prompt=False)
            text = text[: -len("<|im_end|>") - 1]
            self._dbg(f"[PREFIX FRAME 0 TEMPLATE LEN] {len(text)} chars")
            inputs, geometry_inputs = self._frame_processor_inputs(text, images_before[0])
            self._dbg(f"[PREFIX FRAME PROCESSOR OUTPUT] frame_idx=0 {self._tensor_shape_summary(inputs)}")
            _forward_inputs(inputs, geometry_inputs)
            _forward_tokens([self.frame_interval_token_id])

            for frame_idx, frame in enumerate(images_before[1:], start=1):
                frame_text = _image_text(frame_idx)
                self._dbg(f"[PREFIX FRAME TEXT] frame_idx={frame_idx} text={frame_text!r}")
                inputs, geometry_inputs = self._frame_processor_inputs(frame_text, frame)
                self._dbg(f"[PREFIX FRAME PROCESSOR OUTPUT] frame_idx={frame_idx} {self._tensor_shape_summary(inputs)}")
                _forward_inputs(inputs, geometry_inputs)
                _forward_tokens([self.frame_interval_token_id])

            query_text = f"{prompt}\n{_image_text(num_images_before_query)}" if self.stream_frame_timestamps else f"{prompt}<|vision_start|><|image_pad|><|vision_end|>"
            inputs, geometry_inputs = self._frame_processor_inputs(query_text, first_image_after)
            self._dbg(f"[QUERY FRAME PROCESSOR OUTPUT] {self._tensor_shape_summary(inputs)}")
            logits = _forward_inputs(inputs, geometry_inputs)
        else:
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _prompt_first_image_text(0)},
                        {"type": "image", "image": first_image_after},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            text = text[: -len("<|im_end|>") - 1]
            self._dbg(f"[INITIAL CHAT TEMPLATE LEN] {len(text)} chars")
            self._dbg(f"[INITIAL CHAT TEMPLATE]\n{text}")
            inputs, geometry_inputs = self._frame_processor_inputs(text, first_image_after)
            self._dbg(f"[INITIAL PROCESSOR OUTPUT] {self._tensor_shape_summary(inputs)}")
            logits = _forward_inputs(inputs, geometry_inputs)

        next_token = self._sample_next_token(logits)
        sampled_token_id = next_token.item()
        next_token_id = _chosen_decision_token(sampled_token_id, bool(remaining_images_after))
        self._dbg(
            f"[FRAME DECISION] frame_idx=0 after_query_absolute_idx={num_images_before_query} "
            f"sampled_token_id={sampled_token_id} sampled_token={self.tokenizer.decode([sampled_token_id])!r} "
            f"token_id={next_token_id} token={self.tokenizer.decode([next_token_id])!r} "
            f"decision={'CONTINUE' if next_token_id == self.frame_interval_token_id else 'STOP'}"
        )
        _forward_tokens([next_token_id])

        for frame_idx, frame in enumerate(remaining_images_after, start=1):
            if self.stream_frame_policy == "auto" and next_token_id != self.frame_interval_token_id:
                break
            frame_text = _image_text(num_images_before_query + frame_idx)
            self._dbg(f"[FRAME TEXT] frame_idx={frame_idx} absolute_idx={num_images_before_query + frame_idx} text={frame_text!r}")
            inputs, geometry_inputs = self._frame_processor_inputs(frame_text, frame)
            self._dbg(
                f"[FRAME PROCESSOR OUTPUT] frame_idx={frame_idx} "
                f"{self._tensor_shape_summary(inputs)} geometry={tuple(geometry_inputs.shape)}:{geometry_inputs.dtype}:{geometry_inputs.device}"
            )
            logits = _forward_inputs(inputs, geometry_inputs)
            next_token = self._sample_next_token(logits)
            sampled_token_id = next_token.item()
            has_more_frames = frame_idx < len(remaining_images_after)
            next_token_id = _chosen_decision_token(sampled_token_id, has_more_frames)
            self._dbg(
                f"[FRAME DECISION] frame_idx={frame_idx} after_query_absolute_idx={num_images_before_query + frame_idx} "
                f"sampled_token_id={sampled_token_id} sampled_token={self.tokenizer.decode([sampled_token_id])!r} "
                f"token_id={next_token_id} token={self.tokenizer.decode([next_token_id])!r} "
                f"decision={'CONTINUE' if next_token_id == self.frame_interval_token_id else 'STOP'}"
            )
            _forward_tokens([next_token_id])

        chat_prompt_tokens = [
            self.im_end_token_id,
            self.newline_token_id,
            self.im_start_token_id,
        ] + self.assistant_token_ids + [self.newline_token_id]
        self._dbg(f"[FINAL CHAT PROMPT TOKENS] {chat_prompt_tokens}")
        _forward_tokens(chat_prompt_tokens)

        generated_tokens: List[int] = []
        for step in range(max_new_tokens):
            if step == 0:
                last_token = torch.tensor([[chat_prompt_tokens[-1]]], dtype=torch.long, device=self.device)
                cache_position = torch.tensor([current_seq_len - 1], device=self.device)
            else:
                last_token = next_token
                cache_position = torch.tensor([current_seq_len], device=self.device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out = model_to_use(
                    input_ids=last_token,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )
                next_token = self._sample_next_token(out.logits[:, -1, :])
                past_key_values = out.past_key_values

            next_token_id = next_token.item()
            generated_tokens.append(next_token_id)
            current_seq_len += 1
            if next_token_id in {self.eot_token_id, self.im_end_token_id}:
                break

        self._dbg(f"[GENERATED TOKEN IDS] {generated_tokens}")
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        token_count = len(generated_tokens)
        past_key_values = None
        generated_tokens = None
        return text, token_count

    def _history_key(self, doc: Dict[str, Any]) -> Tuple[str, str]:
        trajectory = str(doc.get("trajectory_id") or doc.get("source_video_id") or doc.get("doc_id") or doc.get("id") or "").strip()
        return trajectory, str(doc.get("step") or "").strip()

    def _prepare_doc(self, task: str, doc: Dict[str, Any]) -> Dict[str, Any]:
        if task != "oos_videoqa":
            return doc
        doc = dict(doc)
        if self.oos_history_mode == "none":
            doc["history_messages"] = []
        elif self.oos_history_mode == "pred" and doc.get("mode") == "multi_turn":
            trajectory, _ = self._history_key(doc)
            history = []
            for step in doc.get("depends_on_steps") or []:
                answer = self._oos_pred_history.get((trajectory, str(step)))
                if answer:
                    history.extend([
                        {"role": "user", "content": [{"type": "text", "text": f"Previous step {step}."}]},
                        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                    ])
            doc["history_messages"] = history
        return doc

    def _record_prediction(self, task: str, doc: Dict[str, Any], answer: str) -> None:
        if task == "oos_videoqa" and self.oos_history_mode == "pred":
            key = self._history_key(doc)
            if key[0] and key[1]:
                self._oos_pred_history[key] = answer.strip()

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        def _collate(x):
            return x[2], x[2]

        re_ords = utils.Collator([req.args for req in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))
        pbar = tqdm(total=len(chunks), disable=(self.rank != 0), desc="Stream3D-VLM Responding")
        results: List[GenerationResult] = []
        total_tokens = 0
        total_time = 0.0

        for chunk in chunks:
            self._dbg(f"\n[BATCH] size={len(chunk)}")
            _, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task_name, split_name, idx = task[0], split[0], doc_id[0]
            doc = self._prepare_doc(task_name, self.task_dict[task_name][split_name][idx])
            messages = ChatMessages(**{"messages": doc_to_messages[0](doc)})
            self._dbg("\n" + "=" * 100)
            self._dbg(
                f"[STEP] task={task_name} split={split_name} doc_id={idx} "
                f"step={doc.get('step')} id={doc.get('id')} mode={doc.get('mode')}"
            )
            self._dbg(f"[QUESTION] {doc.get('question')}")
            self._dbg(f"[SOURCE] doc_to_messages={getattr(doc_to_messages[0], '__name__', '')}")
            self._dbg(f"[DOC KEYS] {sorted(doc.keys())}")

            answer, tokens, elapsed = "", 0, 0.0
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats(self.device if self.device.type == "cuda" else None)
                except Exception:
                    pass
            eval_logger.info(f"Stream3D memory before doc={doc.get('id', idx)}: {self._memory_summary()}")
            try:
                answer, tokens, elapsed = self._generate(messages, dict(all_gen_kwargs[0]), doc if task_name == "oos_videoqa" else None)
                eval_logger.info(
                    f"Stream3D doc={doc.get('id', idx)} step={doc.get('step')} "
                    f"tokens={tokens} speed={tokens / elapsed if elapsed > 0 else 0.0:.2f} tok/s"
                )
            except Exception as exc:
                eval_logger.exception(f"Stream3D-VLM generation failed for doc={doc.get('id', idx)}: {exc}")
                if os.getenv("OOS_STRICT_ERRORS", "0") == "1":
                    raise
            finally:
                self._cleanup_memory(f"doc={doc.get('id', idx)}")

            self._record_prediction(task_name, doc, answer)
            total_tokens += tokens
            total_time += elapsed
            results.append(GenerationResult(text=answer, token_counts=TokenCounts(output_tokens=tokens)))
            cache_key = (
                task_name,
                split_name,
                doc.get("id", idx),
                doc.get("video_path"),
                doc.get("query_time_sec"),
                self._prompt_text(messages),
                all_gen_kwargs[0],
            )
            self.cache_hook.add_partial("generate_until", cache_key, answer)
            pbar.update(1)

        pbar.close()
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_time,
            avg_speed=total_tokens / total_time if total_time > 0 else 0,
            additional_metrics={"rank": self.rank},
        )
        return re_ords.get_original(results)

    def loglikelihood(self, requests: List[Instance]):
        raise NotImplementedError("Loglikelihood is not implemented for Stream3D-VLM.")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Multi-round generation is handled through oos_history_mode.")

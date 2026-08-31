import os
import subprocess
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig

# VLM-3R targets older Transformers where these helpers lived in modeling_utils.
try:
    import transformers.modeling_utils as _tf_modeling_utils

    def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        if chunk_size > 0:
            tensor_shape = input_tensors[0].shape[chunk_dim]
            if any(tensor.shape[chunk_dim] != tensor_shape for tensor in input_tensors):
                raise ValueError("All input tensors must have the same shape on chunk_dim")
            if tensor_shape % chunk_size != 0:
                raise ValueError("The dimension to be chunked must be a multiple of chunk_size")
            num_chunks = tensor_shape // chunk_size
            input_chunks = tuple(tensor.chunk(num_chunks, dim=chunk_dim) for tensor in input_tensors)
            return torch.cat(tuple(forward_fn(*chunk) for chunk in zip(*input_chunks)), dim=chunk_dim)
        return forward_fn(*input_tensors)

    def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask), dtype=torch.long)[mask].long()
        return heads, index

    if not hasattr(_tf_modeling_utils, "apply_chunking_to_forward"):
        _tf_modeling_utils.apply_chunking_to_forward = _apply_chunking_to_forward
    if not hasattr(_tf_modeling_utils, "find_pruneable_heads_and_indices"):
        _tf_modeling_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
except Exception:
    pass

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.vlm_3r_media import MediaItem, extract_ordered_media_and_turns

try:
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except ImportError as exc:
    raise ImportError(
        "VLM-3R requires the VITA-Group/VLM-3R repository on PYTHONPATH. "
        "Add the pinned VLM-3R source tree to PYTHONPATH before launching."
    ) from exc


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@register_model("vlm_3r")
class VLM3R(lmms):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "Journey9ni/vlm-3r-llava-qwen2-lora",
        model_base: str = "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2",
        conv_mode: str = "qwen_1_5",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        for_get_frames_num: int = 32,
        mm_spatial_pool_stride: int = 2,
        mm_spatial_pool_mode: str = "bilinear",
        mm_resampler_type: str = "spatial_pool",
        mm_spatial_pool_out_channels: int = 1024,
        mm_pooling_position: str = "after",
        delay_load: bool = False,
        mm_newline_position: str = "grid",
        load_8bit: bool = False,
        load_4bit: bool = False,
        low_cpu_mem_usage: bool = True,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "float16",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.1,
        num_beams: int = 1,
        disable_cudnn: bool = False,
        system_prompt: str = "",
        add_time_instruction: bool = False,
        export_point_cloud: bool = False,
        point_cloud_output_dir: str = "point_clouds",
        point_cloud_export_limit: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.path = pretrained
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            raise ValueError("VLM-3R wrapper currently supports batch_size=1 only.")

        self.for_get_frames_num = int(for_get_frames_num)
        self.mm_resampler_type = mm_resampler_type
        self.mm_spatial_pool_stride = int(mm_spatial_pool_stride)
        self.mm_spatial_pool_out_channels = int(mm_spatial_pool_out_channels)
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.mm_newline_position = mm_newline_position
        self.mm_resampler_location = mm_pooling_position
        self.delay_load = self._as_bool(delay_load)
        self.low_cpu_mem_usage = self._as_bool(low_cpu_mem_usage)
        self.default_max_new_tokens = int(max_new_tokens)
        self.default_temperature = float(temperature)
        self.default_top_p = float(top_p)
        self.default_num_beams = int(num_beams)
        self.system_prompt = self._resolve_system_prompt(system_prompt)
        self.add_time_instruction = self._as_bool(add_time_instruction)
        self.export_point_cloud = self._as_bool(export_point_cloud)
        self.point_cloud_output_dir = point_cloud_output_dir
        self.point_cloud_export_limit = int(point_cloud_export_limit)
        self._point_cloud_exports = 0
        if self.export_point_cloud:
            os.makedirs(self.point_cloud_output_dir, exist_ok=True)
        self.disable_cudnn = self._as_bool(disable_cudnn)
        if self.disable_cudnn:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            eval_logger.warning("VLM-3R cuDNN was explicitly disabled by disable_cudnn=True.")
        else:
            torch.backends.cudnn.enabled = True
            eval_logger.info(f"VLM-3R cuDNN enabled (version={torch.backends.cudnn.version()}).")

        self.accelerator = Accelerator()
        if self.accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{self.accelerator.local_process_index}")
            self.device_map = f"cuda:{self.accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

        model_name = get_model_name_from_path(pretrained)
        overwrite_config = {
            "mm_resampler_type": self.mm_resampler_type,
            "mm_spatial_pool_stride": self.mm_spatial_pool_stride,
            "mm_spatial_pool_out_channels": self.mm_spatial_pool_out_channels,
            "mm_spatial_pool_mode": self.mm_spatial_pool_mode,
            "mm_pooling_position": self.mm_resampler_location,
            "mm_newline_position": self.mm_newline_position,
            "add_faster_video": False,
            "delay_load": self.delay_load,
        }

        self.cfg_pretrained = AutoConfig.from_pretrained(pretrained)
        load_8bit_enabled = self._as_bool(load_8bit)
        load_4bit_enabled = self._as_bool(load_4bit)
        self._tokenizer, self._model, self.image_processor, self.context_len = load_pretrained_model(
            pretrained,
            model_base,
            model_name,
            load_8bit=load_8bit_enabled,
            load_4bit=load_4bit_enabled,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            overwrite_config=overwrite_config,
        )
        self._model.eval()
        if getattr(self._model, "hf_device_map", None) is None:
            if load_8bit_enabled or load_4bit_enabled:
                self._model.to(self._device)
            else:
                dtype_by_name = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }
                target_dtype = torch_dtype if isinstance(torch_dtype, torch.dtype) else dtype_by_name.get(str(torch_dtype).lower())
                if target_dtype is None:
                    raise ValueError(f"Unsupported VLM-3R torch_dtype: {torch_dtype}")
                # assign=True in the low-memory loader preserves checkpoint
                # dtypes. Apply the requested inference dtype uniformly so the
                # vision, spatial, and fusion modules agree at runtime.
                self._model.to(device=self._device, dtype=target_dtype)
                eval_logger.info(f"VLM-3R moved to {self._device} with dtype={target_dtype}.")

        if self._tokenizer.pad_token_id is None and "qwen" in getattr(self._tokenizer, "name_or_path", "").lower():
            self._tokenizer.pad_token_id = 151643

        spatial_tower = self.model.get_model().get_spatial_tower() if hasattr(self.model, "get_model") else None
        if spatial_tower is not None and hasattr(spatial_tower, "config"):
            spatial_tower.config.export_point_cloud = self.export_point_cloud
            spatial_tower.config.point_cloud_output_dir = self.point_cloud_output_dir

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def _debug_enabled(self) -> bool:
        return os.getenv("OOS_CHAT_DEBUG", "0") == "1"

    def _dbg(self, msg: str) -> None:
        if self._debug_enabled() and self.rank == 0:
            print(msg, flush=True)

    def _safe_filename_part(self, value: Any) -> str:
        text = str(value if value is not None else "unknown")
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:160]

    def _point_cloud_output_paths(self, doc: Dict[str, Any], doc_id: Any) -> Optional[List[str]]:
        if not self.export_point_cloud:
            return None
        if self.point_cloud_export_limit > 0 and self._point_cloud_exports >= self.point_cloud_export_limit:
            return None
        traj_id = self._safe_filename_part(doc.get("trajectory_id", doc.get("source_video_id", doc.get("id", doc_id))))
        step_id = self._safe_filename_part(doc.get("step", doc_id))
        rank_dir = os.path.join(self.point_cloud_output_dir, f"rank_{self.rank}")
        os.makedirs(rank_dir, exist_ok=True)
        self._point_cloud_exports += 1
        return [os.path.join(rank_dir, f"{traj_id}_step_{step_id}_doc_{doc_id}.ply")]

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _load_video(self, video_path: str) -> Tuple[np.ndarray, str, float]:
        if VideoReader is not None:
            return self._load_video_decord(video_path)

        return self._load_video_ffmpeg(video_path)

    def _load_video_decord(self, video_path: str) -> Tuple[np.ndarray, str, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = float(vr.get_avg_fps())
        video_time = total_frame_num / avg_fps if avg_fps > 0 else 0.0
        fps = max(round(avg_fps), 1)

        frame_idx = [i for i in range(0, total_frame_num, fps)]
        frame_time = [i / fps for i in frame_idx]

        if self.for_get_frames_num > 0 and len(frame_idx) > self.for_get_frames_num:
            frame_idx = np.linspace(0, total_frame_num - 1, self.for_get_frames_num, dtype=int).tolist()
            frame_time = [i / avg_fps for i in frame_idx]

        spare_frames = vr.get_batch(frame_idx).asnumpy()
        frame_time_text = ",".join([f"{i:.2f}s" for i in frame_time])
        return spare_frames, frame_time_text, video_time

    def _probe_video_duration(self, video_path: str) -> float:
        ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")
        ffprobe_path = os.getenv("FFPROBE_PATH") or ffmpeg_path.replace("ffmpeg", "ffprobe")
        try:
            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _load_video_ffmpeg(self, video_path: str) -> Tuple[np.ndarray, str, float]:
        ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")
        video_time = self._probe_video_duration(video_path)

        with tempfile.TemporaryDirectory(prefix="vlm3r_frames_") as tmpdir:
            frame_pattern = os.path.join(tmpdir, "frame_%06d.jpg")
            cmd = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                video_path,
                "-vf",
                "fps=1",
                frame_pattern,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            frame_paths = sorted(
                os.path.join(tmpdir, name)
                for name in os.listdir(tmpdir)
                if name.lower().endswith(".jpg")
            )
            if not frame_paths:
                raise RuntimeError(f"ffmpeg extracted no frames from video: {video_path}")

            selected_idx = list(range(len(frame_paths)))
            if self.for_get_frames_num > 0 and len(selected_idx) > self.for_get_frames_num:
                selected_idx = np.linspace(0, len(frame_paths) - 1, self.for_get_frames_num, dtype=int).tolist()

            frames = []
            frame_time = []
            for idx in selected_idx:
                with Image.open(frame_paths[idx]) as image:
                    frames.append(np.array(image.convert("RGB")))
                frame_time.append(float(idx))

        frame_time_text = ",".join([f"{i:.2f}s" for i in frame_time])
        return np.stack(frames, axis=0), frame_time_text, video_time

    def _text_from_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return str(content).strip()
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and str(part.get("text", "")).strip()
        ).strip()

    def _normalise_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalised = []
        for msg in messages:
            item = dict(msg)
            content = item.get("content", [])
            if isinstance(content, str):
                item["content"] = [{"type": "text", "text": content}]
            elif content is None:
                item["content"] = []
            else:
                item["content"] = [dict(part) if isinstance(part, dict) else {"type": "text", "text": str(part)} for part in content]
            normalised.append(item)
        return normalised

    def _visual_content_from_task(self, doc_to_messages, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        bound_task = getattr(doc_to_messages, "__self__", None)
        if bound_task is None or not hasattr(bound_task, "doc_to_visual"):
            return []

        visuals = bound_task.doc_to_visual(doc)
        visual_content = []
        for visual in visuals:
            if isinstance(visual, str) and visual.lower().endswith(VIDEO_EXTS):
                visual_content.append({"type": "video", "url": visual})
            elif isinstance(visual, str) and visual.lower().endswith(IMAGE_EXTS):
                visual_content.append({"type": "image", "url": visual})
            else:
                visual_content.append({"type": "image", "url": visual})
        return visual_content

    def _visual_token(self) -> str:
        if self.model.config.mm_use_im_start_end:
            return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        return DEFAULT_IMAGE_TOKEN

    def _extract_media_and_prompt(self, messages: List[Dict[str, Any]]) -> Tuple[List[MediaItem], List[Tuple[str, str]], str]:
        """Extract ordered media and keep matching visual tokens in each turn."""
        return extract_ordered_media_and_turns(messages, self._visual_token())

    def _prepare_generation_inputs(self, media: List[MediaItem], question: Any, system_prompt: str = ""):
        tensors: List[torch.Tensor] = []
        modalities: List[str] = []
        time_instruction = ""
        video_count = sum(modality == "video" for modality, _ in media)
        if video_count > 1:
            raise NotImplementedError(f"VLM-3R inference supports one video per request, got {video_count}.")

        for modality, path in media:
            if modality == "video":
                video, frame_time, video_time = self._load_video(path)
                if self.add_time_instruction:
                    time_instruction = (
                        f"The video lasts for {video_time:.2f} seconds, and {len(video)} frames "
                        f"are uniformly sampled from it. These frames are located at {frame_time}. "
                        "Please answer the following questions related to this video."
                    )
                tensor = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
                tensors.append(tensor.half().to(self.device))
                modalities.append("video")
            elif modality == "image":
                with Image.open(path) as image:
                    pil_image = image.convert("RGB")
                tensor = self.image_processor.preprocess([pil_image], return_tensors="pt")["pixel_values"]
                if tensor.shape[0] != 1:
                    raise ValueError(f"Expected one processed image for {path}, got shape {tuple(tensor.shape)}.")
                tensors.append(tensor[0].half().to(self.device))
                modalities.append("image")
            else:
                raise ValueError(f"Unsupported VLM-3R modality: {modality!r}")

        has_media = bool(media)

        if isinstance(question, list):
            turns = [(role, text) for role, text in question if text]
        else:
            turns = [("user", str(question))]

        visual_token = self._visual_token()
        visual_token_count = sum(text.count(DEFAULT_IMAGE_TOKEN) for _, text in turns)
        if has_media and visual_token_count == 0:
            for turn_idx, (role, text) in enumerate(turns):
                if role == "user":
                    prefix = "\n".join([visual_token] * len(media))
                    turns[turn_idx] = (role, f"{prefix}\n{text}")
                    visual_token_count = len(media)
                    break
        if visual_token_count != len(tensors):
            raise ValueError(
                "VLM-3R visual token/media mismatch: "
                f"prompt has {visual_token_count} visual tokens but {len(tensors)} media tensors."
            )

        conv = conv_templates[self.conv_mode].copy()
        if system_prompt:
            if conv.sep_style == SeparatorStyle.CHATML and not system_prompt.lstrip().startswith("<|im_start|>system"):
                conv.system = f"<|im_start|>system\n{system_prompt}"
            else:
                conv.system = system_prompt
        time_instruction_added = False
        for role, text in turns:
            conv_role = conv.roles[1] if role == "assistant" else conv.roles[0]
            msg_text = text
            if role == "user":
                if time_instruction and not time_instruction_added and DEFAULT_IMAGE_TOKEN in msg_text:
                    msg_text = f"{time_instruction}\n{msg_text}"
                    time_instruction_added = True
            conv.append_message(conv_role, msg_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if has_media:
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        return input_ids, attention_mask, tensors, modalities, stop_str, stopping_criteria, prompt

    def _messages_have_media(self, messages: List[Dict[str, Any]]) -> bool:
        return any(
            isinstance(part, dict) and part.get("type") in {"video", "image"}
            for msg in messages
            for part in msg.get("content", [])
        )

    def _build_messages(self, doc_to_messages, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = self._normalise_messages(doc_to_messages(doc))
        if self.system_prompt:
            messages = self._apply_system_prompt(messages, self.system_prompt)

        source_name = getattr(doc_to_messages, "__name__", "")
        has_media = self._messages_have_media(messages)

        if source_name == "oos_doc_to_messages_with_visuals":
            if not has_media:
                if os.getenv("OOS_NO_VIDEO_INPUT", "0") == "1":
                    self._dbg("[VLM-3R] OOS_NO_VIDEO_INPUT=1; using text-only messages")
                    return messages
                raise ValueError(
                    "oos_doc_to_messages_with_visuals was selected, but it returned no video/image content."
                )
            self._dbg("[VLM-3R] using visuals embedded by oos_doc_to_messages_with_visuals")
            return messages

        if not has_media:
            visual_content = self._visual_content_from_task(doc_to_messages, doc)
            if visual_content:
                last_user = None
                for idx in range(len(messages) - 1, -1, -1):
                    if messages[idx].get("role") == "user":
                        last_user = idx
                        break
                if last_user is None:
                    messages.append({"role": "user", "content": visual_content})
                else:
                    messages[last_user]["content"] = visual_content + messages[last_user].get("content", [])
        return messages

    def _init_pred_history(self) -> None:
        if not hasattr(self, "_pred_history"):
            self._pred_history = defaultdict(dict)

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res: List[GenerationResult] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        pred_mode = os.getenv("OOS_HISTORY_MODE", "gold").strip().lower() == "pred"

        total_elapsed_time = 0.0
        total_tokens = 0

        for request in requests:
            ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            messages = self._build_messages(doc_to_messages, doc)

            if pred_mode:
                self._init_pred_history()
                traj_id = str(doc.get("trajectory_id", doc.get("source_video_id", doc.get("id"))))
                deps = [str(x) for x in doc.get("depends_on_steps", [])]
                generated_history = []
                for dep_step in deps:
                    if dep_step not in self._pred_history[traj_id]:
                        continue
                    q_text, a_text = self._pred_history[traj_id][dep_step]
                    generated_history.append({"role": "user", "content": [{"type": "text", "text": q_text}]})
                    generated_history.append({"role": "assistant", "content": [{"type": "text", "text": a_text}]})
                if generated_history:
                    system_msgs = [m for m in messages if m.get("role") == "system"]
                    non_system = [m for m in messages if m.get("role") != "system"]
                    messages = system_msgs + generated_history + non_system

            media, question, system_prompt = self._extract_media_and_prompt(messages)
            input_ids, attention_mask, visual_tensors, modalities, stop_str, stopping_criteria, prompt = self._prepare_generation_inputs(
                media,
                question or str(ctx),
                system_prompt,
            )
            self._dbg(
                f"[VLM-3R MEDIA] ordered_media={media} modalities={modalities} "
                f"tensor_shapes={[tuple(tensor.shape) for tensor in visual_tensors]}"
            )

            request_gen_kwargs = dict(gen_kwargs or {})
            max_new_tokens = int(request_gen_kwargs.get("max_new_tokens", self.default_max_new_tokens))
            temperature = float(request_gen_kwargs.get("temperature", self.default_temperature))
            do_sample = temperature > 0

            generate_args = {
                "inputs": input_ids,
                "attention_mask": attention_mask,
                "do_sample": do_sample,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": float(request_gen_kwargs.get("top_p", self.default_top_p)),
                "num_beams": int(request_gen_kwargs.get("num_beams", self.default_num_beams)),
                "use_cache": True,
                "stopping_criteria": [stopping_criteria],
            }
            if visual_tensors:
                generate_args["images"] = visual_tensors
                generate_args["modalities"] = modalities
                point_cloud_output_paths = self._point_cloud_output_paths(doc, doc_id)
                if point_cloud_output_paths:
                    generate_args["point_cloud_output_paths"] = point_cloud_output_paths
                    self._dbg(f"[VLM-3R] exporting point cloud to {point_cloud_output_paths[0]}")

            start_time = time.time()
            with torch.inference_mode():
                output_ids = self.model.generate(**generate_args)
            end_time = time.time()

            if output_ids.shape[-1] > input_ids.shape[-1]:
                generated_ids = output_ids[:, input_ids.shape[-1] :]
                output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                if not output_text:
                    generated_ids = output_ids
                    output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            else:
                generated_ids = output_ids
                output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if output_text.endswith(stop_str):
                output_text = output_text[: -len(stop_str)].strip()

            output_tokens = int(generated_ids.shape[-1])
            total_tokens += output_tokens
            total_elapsed_time += end_time - start_time

            if pred_mode:
                self._init_pred_history()
                traj_id = str(doc.get("trajectory_id", doc.get("source_video_id", doc.get("id"))))
                step_id = str(doc.get("step"))
                self._pred_history[traj_id][step_id] = (str(doc.get("question", "")).strip(), output_text)

            self._dbg(f"[VLM-3R PROMPT]\n{prompt}")
            self._dbg(f"[VLM-3R OUTPUT]\n{output_text}")

            res.append(GenerationResult(text=output_text, token_counts=TokenCounts(output_tokens=output_tokens)))
            self.cache_hook.add_partial("generate_until", (prompt, request_gen_kwargs), output_text)
            pbar.update(1)

        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0,
            additional_metrics={"rank": self.rank},
        )
        pbar.close()
        return res

    def loglikelihood(self, requests):
        raise NotImplementedError("VLM-3R chat wrapper only supports generation.")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("VLM-3R chat wrapper only supports generate_until.")

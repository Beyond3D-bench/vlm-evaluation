from __future__ import annotations

import os
import json
from datetime import timedelta
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages

try:
    from cambrian.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from cambrian.conversation import conv_templates
    from cambrian.mm_utils import (
        expand2square,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from cambrian.model.builder import load_pretrained_model

    _CAMBRIAN_IMPORT_ERROR = None
except ImportError as exc:
    DEFAULT_IM_END_TOKEN = "<im_end>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200
    conv_templates = None
    expand2square = None
    get_model_name_from_path = None
    process_images = None
    tokenizer_image_token = None
    load_pretrained_model = None
    _CAMBRIAN_IMPORT_ERROR = exc


def _require_cambrian() -> None:
    if _CAMBRIAN_IMPORT_ERROR is not None:
        raise ImportError(
            "Cambrian-P requires the Cambrian package. Install the Cambrian/Cambrian-S "
            "package, for example: pip install git+https://github.com/cambrian-mllm/cambrian-s.git"
        ) from _CAMBRIAN_IMPORT_ERROR


def _oos_debug_line(label: str, **fields: Any) -> None:
    if os.getenv("OOS_CHAT_DEBUG", "0") != "1":
        return
    payload = json.dumps(fields, ensure_ascii=True, default=str)
    print(f"{label}: {payload}", flush=True)


def is_video_file(file_path: Any) -> bool:
    if isinstance(file_path, Image.Image) or not isinstance(file_path, str):
        return False
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions


def is_image_file(file_path: Any) -> bool:
    if isinstance(file_path, Image.Image):
        return True
    if not isinstance(file_path, str):
        return False
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions


def process_video_with_decord(video_file, model_cfg, num_threads=-1):
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError("Cambrian-P video input requires decord. Install it with: pip install decord") from exc

    if num_threads < 1:
        vr = VideoReader(video_file, ctx=cpu(0))
    else:
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=num_threads)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / model_cfg.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    frame_time = [i / avg_fps for i in frame_idx]

    if model_cfg.video_max_frames > 0:
        if len(frame_idx) > model_cfg.video_max_frames or model_cfg.video_force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, model_cfg.video_max_frames, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]

    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    num_frames_to_sample = len(frame_idx)
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample


def process_videos(videos, image_processor, model_cfg, num_threads=-1):
    processor_aux_list = image_processor
    new_videos_aux_list = []
    video_sizes = []

    for video in videos:
        video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video, model_cfg, num_threads=num_threads)
        video_sizes.append((video.shape[2], video.shape[1], video.shape[0]))
        video = [Image.fromarray(video[_], mode="RGB") for _ in range(video.shape[0])]

        video_aux_list = []
        for processor_aux in processor_aux_list:
            video_aux = [expand2square(image, tuple(int(x * 255) for x in processor_aux.image_mean)) for image in video]
            video_aux_list.append(processor_aux.preprocess(video_aux, return_tensors="pt")["pixel_values"])

        new_videos_aux_list.append(video_aux_list)

    new_videos_aux_list = [list(batch_video_aux) for batch_video_aux in zip(*new_videos_aux_list)]
    new_videos_aux_list = [torch.stack(video_aux) for video_aux in new_videos_aux_list]
    return new_videos_aux_list, video_sizes, (video_time, frame_time, num_frames_to_sample)


@register_model("cambrian_p_chat")
class CambrianP(lmms):
    """
    Cambrian-P chat wrapper.

    Cambrian-P is loaded through the Cambrian model builder and uses the same
    multimodal tensor path as the Cambrian-S wrappers. The difference from the
    simple Cambrian-S backend is that this class consumes task doc_to_messages
    directly, preserving chat history and typed image/video content.
    """

    is_simple = False

    def __init__(
        self,
        pretrained: str = "nyu-visionx/Cambrian-P-8B",
        torch_dtype: Optional[Union[str, torch.dtype]] = "float16",
        batch_size: Optional[Union[int, str]] = 1,
        device_map: str = "cuda:0",
        conv_template: str = "qwen_2",
        use_cache: bool = True,
        truncate_context: bool = False,
        video_max_frames: int = 32,
        video_fps: int = 1,
        video_force_sample: bool = False,
        add_time_instruction: bool = False,
        miv_token_len: int = 196,
        si_token_len: int = 729,
        image_aspect_ratio: str = "anyres",
        anyres_max_subimages: int = 9,
        **kwargs,
    ) -> None:
        super().__init__()
        _require_cambrian()
        loader_kwargs = dict(kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and (device_map == "auto" or device_map == "balanced_low_0"):
            raise NotImplementedError("device_map == auto is not supported for cambrian_p yet.")
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.torch_dtype = torch_dtype
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            eval_logger.warning("Cambrian-P chat currently builds one multimodal sample at a time. Forcing batch_size=1.")
            self.batch_size_per_gpu = 1

        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained,
            None,
            self.model_name,
            device_map=self.device_map,
            **loader_kwargs,
        )

        self._model.config.video_max_frames = video_max_frames
        self._model.config.video_fps = video_fps
        self._model.config.video_force_sample = video_force_sample
        self._model.config.add_time_instruction = add_time_instruction
        self._model.config.connector_only = getattr(self._model.config, "connector_only", True)
        self._model.config.miv_token_len = miv_token_len
        self._model.config.si_token_len = si_token_len
        self._model.config.image_aspect_ratio = image_aspect_ratio
        self._model.config.anyres_max_subimages = anyres_max_subimages

        self._config = self._model.config
        self.model.eval()

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP, FSDP, and DeepSpeed are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                ds_kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **ds_kwargs)
                eval_logger.info("Detected DeepSpeed. Make sure accelerate is configured with ZeRO stage 0.")
            if accelerator.distributed_type in {DistributedType.FSDP, DistributedType.DEEPSPEED}:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

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

    @staticmethod
    def _content_url(content: Any) -> Any:
        return getattr(content, "url", None)

    def _media_token(self) -> str:
        if getattr(self._config, "mm_use_im_start_end", False):
            return f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
        return DEFAULT_IMAGE_TOKEN

    def _normalise_image(self, image: Any) -> Any:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image

    def _message_text_and_media(self, chat_messages: ChatMessages) -> Tuple[List[dict], List[Any]]:
        system_parts: List[str] = []
        turns: List[dict] = []
        media: List[Any] = []

        for message in chat_messages.messages:
            text_parts: List[str] = []
            for content in message.content:
                ctype = getattr(content, "type", None)
                if ctype == "text":
                    text = getattr(content, "text", "")
                    if text:
                        text_parts.append(text)
                elif ctype in {"image", "video"}:
                    media_item = self._content_url(content)
                    media.append(media_item)
                    text_parts.append(self._media_token())
                elif ctype == "audio":
                    eval_logger.warning("Cambrian-P chat does not support audio content; dropping audio item.")

            text = "\n".join(part for part in text_parts if part)
            if not text:
                continue
            if message.role == "system":
                system_parts.append(text)
            elif message.role in {"user", "assistant"}:
                turns.append({"role": message.role, "text": text})
            else:
                raise ValueError(f"Unsupported chat role for Cambrian-P: {message.role}")

        if system_parts:
            system_text = "\n".join(system_parts)
            if turns and turns[0]["role"] == "user":
                turns[0]["text"] = f"{system_text}\n{turns[0]['text']}"
            else:
                turns.insert(0, {"role": "user", "text": system_text})

        return turns, media

    def _process_media(self, media: List[Any]):
        if not media:
            return None, None

        if all(is_image_file(item) for item in media):
            images = [self._normalise_image(item) for item in media]
            return process_images(images, self._image_processor, self._config, use_pad=True)

        if len(media) == 1 and is_video_file(media[0]):
            visual = media[0]
            num_threads = 1 if isinstance(visual, str) and ("Ego4D" in visual or "video_mmmu" in visual) else -1
            tensors, sizes, _ = process_videos(
                [visual],
                self._image_processor,
                self._config,
                num_threads=num_threads,
            )
            return tensors, sizes

        if sum(is_video_file(item) for item in media) > 1:
            raise NotImplementedError("Cambrian-P chat does not support multiple videos in one request.")

        visual_tensors = []
        visual_sizes = []
        for item in media:
            if is_video_file(item):
                num_threads = 1 if isinstance(item, str) and ("Ego4D" in item or "video_mmmu" in item) else -1
                tensors, sizes, _ = process_videos(
                    [item],
                    self._image_processor,
                    self._config,
                    num_threads=num_threads,
                )
                visual_tensors.append(tensors[0])
                visual_sizes.append(sizes[0])
            elif is_image_file(item):
                image = self._normalise_image(item)
                tensors, sizes = process_images([image], self._image_processor, self._config, use_pad=True)
                visual_tensors.append(tensors[0])
                visual_sizes.append(sizes[0])
            else:
                raise NotImplementedError(f"Unsupported Cambrian-P media item: {type(item).__name__} {item!r}")

        return visual_tensors, visual_sizes

    def _build_prompt(self, turns: List[dict]) -> str:
        if not turns:
            raise ValueError("doc_to_messages produced no text turns for Cambrian-P.")

        prompt_parts: List[str] = []
        for turn in turns:
            text = turn.get("text")
            if not text:
                continue
            role = "assistant" if turn.get("role") == "assistant" else "user"
            prompt_parts.append(f"<|im_start|>{role}\n{text}<|im_end|>")

        if turns[-1]["role"] != "assistant":
            prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)

    def _make_one_request(self, request: Instance):
        _, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=raw_messages)
        turns, media = self._message_text_and_media(chat_messages)
        visual_tensors, visual_sizes = self._process_media(media)
        prompt = self._build_prompt(turns)
        tensor_shapes = None
        if visual_tensors is not None:
            tensor_shapes = [tuple(tensor.shape) for tensor in visual_tensors]
        _oos_debug_line(
            "media",
            doc_id=doc_id,
            media=media,
            image_sizes=visual_sizes,
            tensor_shapes=tensor_shapes,
        )
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        return input_ids, visual_tensors, visual_sizes, prompt, dict(gen_kwargs or {})

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Cambrian-P chat does not implement loglikelihood.")

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res: List[GenerationResult] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for request in requests:
            doc_id = request.args[3]
            input_ids, visual_tensors, visual_sizes, prompt, gen_kwargs = self._make_one_request(request)

            until = gen_kwargs.pop("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected gen_kwargs['until'] to be a string or list, got {type(until)}")

            gen_kwargs.setdefault("max_new_tokens", 1024)
            gen_kwargs.setdefault("temperature", 0)
            gen_kwargs.setdefault("top_p", None)
            gen_kwargs.setdefault("num_beams", 1)

            input_ids = input_ids.to(self.device, non_blocking=True)
            if visual_tensors is not None:
                visual_tensors = [tensor.half().to(self.device, non_blocking=True) for tensor in visual_tensors]

            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=visual_tensors,
                    image_sizes=visual_sizes,
                    use_cache=self.use_cache,
                    do_sample=gen_kwargs["temperature"] > 0,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )

            output_tokens = output_ids.shape[-1] if hasattr(output_ids, "shape") else None
            text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            for term in until:
                if term:
                    text = text.split(term)[0]

            if os.getenv("OOS_CHAT_DEBUG", "0") == "1":
                _oos_debug_line("prompt", doc_id=doc_id, prompt=prompt)
                _oos_debug_line("answer", doc_id=doc_id, answer=text)
            else:
                eval_logger.debug(f"Question: {prompt}")
                eval_logger.debug(f"Answer: {text}")
            res.append(GenerationResult(text=text, token_counts=TokenCounts(output_tokens=output_tokens)))
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Cambrian-P chat handles multi-turn prompts through doc_to_messages.")

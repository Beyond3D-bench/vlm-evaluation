import os
import time
import warnings
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    InternVLForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

warnings.filterwarnings("ignore")

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_width_ratio, target_height_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_width_ratio
    target_height = image_size * target_height_ratio
    blocks = target_width_ratio * target_height_ratio
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


@register_model("internvl_hf_chat")
class InternVLHf(lmms):
    """
    InternVL Model for Hugging Face Transformers: https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/internvl
    At present, the OpenGVLab has provided the HF format model weights for InternVL3 and InternVL3.5:
        InternVL3-1B: https://huggingface.co/OpenGVLab/InternVL3-1B-hf
        InternVL3-2B: https://huggingface.co/OpenGVLab/InternVL3-2B-hf
        InternVL3-8B: https://huggingface.co/OpenGVLab/InternVL3-8B-hf
        InternVL3-14B: https://huggingface.co/OpenGVLab/InternVL3-14B-hf
        InternVL3-38B: https://huggingface.co/OpenGVLab/InternVL3-38B-hf
        InternVL3-78B: https://huggingface.co/OpenGVLab/InternVL3-78B-hf

        InternVL3.5-1B: https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF
        InternVL3.5-2B: https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF
        InternVL3.5-4B: https://huggingface.co/OpenGVLab/InternVL3_5-4B-HF
        InternVL3.5-8B: https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF
        InternVL3.5-14B: https://huggingface.co/OpenGVLab/InternVL3_5-14B-HF
        InternVL3.5-38B: https://huggingface.co/OpenGVLab/InternVL3_5-38B-HF
        ...

    Example usage:

    accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
        --model internvl_hf \
        --model_args pretrained=OpenGVLab/InternVL3_5-8B-HF \
        --tasks seedbench \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

    is_simple = False

    def _debug_enabled(self) -> bool:
        return os.getenv("OOS_CHAT_DEBUG", "0") == "1"

    def _dbg(self, msg: str) -> None:
        if self._debug_enabled():
            print(msg, flush=True)

    def _as_bool(self, value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _content_preview(self, content: List[Dict], max_text_chars: int = 2000) -> str:
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
                parts.append(f"image='{item.get('url', item.get('image', ''))}'")
            else:
                parts.append(str(item))
        return " | ".join(parts)

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B-HF",
        revision: str = "main",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        min_patches: int = 1,
        max_patches: int = 12,
        num_frames: int = 32,
        fps: Optional[float] = None,
        do_sample_frames: bool = True,
        do_resize_video: bool = True,
        video_width: int = 448,
        video_height: int = 448,
        trust_remote_code: Optional[bool] = False,
        low_cpu_mem_usage: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        use_cache: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.path = pretrained
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.num_frames = num_frames
        self.fps = fps
        self.do_sample_frames = self._as_bool(do_sample_frames)
        self.do_resize_video = self._as_bool(do_resize_video)
        self.video_width = int(video_width)
        self.video_height = int(video_height)
        self.use_remote_chat = self._as_bool(trust_remote_code)

        batch_size_int = int(batch_size)
        assert batch_size_int == 1, f"Batch size should be 1 for InternVLHf, but got {batch_size_int}."
        self.batch_size_per_gpu = batch_size_int

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "revision": revision,
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "attn_implementation": attn_implementation,
            "trust_remote_code": trust_remote_code,
            "device_map": self.device_map,
        }
        model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}

        load_in_4bit = self._as_bool(load_in_4bit)
        load_in_8bit = self._as_bool(load_in_8bit)
        if load_in_4bit and load_in_8bit:
            raise ValueError("Only one of load_in_4bit or load_in_8bit can be True.")

        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig

            model_kwargs.pop("dtype", None)
            if load_in_4bit:
                compute_dtype = getattr(torch, str(bnb_4bit_compute_dtype), None)
                if compute_dtype is None:
                    raise ValueError(f"Unsupported bnb_4bit_compute_dtype={bnb_4bit_compute_dtype}")
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=self._as_bool(bnb_4bit_use_double_quant),
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self._dbg(f"[MODEL LOAD KWARGS] {model_kwargs}")
        model_cls = AutoModel if self._as_bool(trust_remote_code) else InternVLForConditionalGeneration
        self._model = model_cls.from_pretrained(
            self.path,
            **model_kwargs,
        ).eval()
        self._config = self._model.config

        if self.use_remote_chat:
            self.processor = None
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                use_fast=False,
            )
        else:
            self.processor = AutoProcessor.from_pretrained(
                self.path,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
            self._tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        """Return the model configuration."""
        return self._config

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

    @property
    def model(self):
        """Return the unwrapped model."""
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        """Return the end-of-sentence token ID to replace the end-of-text token ID."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        """Return the padding token ID."""
        return self.tokenizer.pad_token_id

    @property
    def max_length(self):
        """Return the maximum sequence length."""
        return self._max_length

    @property
    def batch_size(self):
        """Return the batch size per GPU."""
        return self.batch_size_per_gpu

    @property
    def device(self):
        """Return the device."""
        return self._device

    @property
    def rank(self):
        """Return the process rank."""
        return self._rank

    @property
    def world_size(self):
        """Return the world size."""
        return self._world_size

    def flatten(self, input):
        """Flatten a nested list."""
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _text_from_content(self, content: List[Dict]) -> str:
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and str(part.get("text", "")).strip()
        ).strip()

    def _remote_chat_parts(self, messages: ChatMessages) -> Tuple[str, List[Tuple[str, str]], str, List[str]]:
        system_parts: List[str] = []
        history: List[Tuple[str, str]] = []
        pending_user: Optional[str] = None
        current_text = ""
        video_paths: List[str] = []

        dumped = messages.model_dump()["messages"]
        for msg in dumped:
            role = msg.get("role")
            content = msg.get("content", [])
            text = self._text_from_content(content)
            if role == "system":
                if text:
                    system_parts.append(text)
                continue
            if role == "user":
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        video_paths.append(item.get("url") or item.get("video"))
                if pending_user is not None:
                    history.append((pending_user, ""))
                pending_user = text
                current_text = text
            elif role == "assistant":
                if pending_user is not None:
                    history.append((pending_user, text))
                    pending_user = None

        if pending_user is None and history:
            current_text = history[-1][0]
            history = history[:-1]
        elif pending_user is not None:
            current_text = pending_user

        return "\n\n".join(system_parts).strip(), history, current_text, [v for v in video_paths if v]

    def _sample_frame_indices(self, video_path: str, vr: VideoReader) -> np.ndarray:
        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        video_fps = float(vr.get_avg_fps())
        if self.fps is not None and self.fps > 0:
            duration = total_frames / max(video_fps, 1e-6)
            num_segments = max(1, int(round(duration * float(self.fps))))
            if self.num_frames is not None:
                num_segments = min(num_segments, int(self.num_frames))
        else:
            num_segments = int(self.num_frames)
        num_segments = max(1, min(num_segments, total_frames))
        return np.linspace(0, total_frames - 1, num_segments, dtype=int)

    def _load_remote_video(self, video_path: str) -> Tuple[torch.Tensor, List[int]]:
        if VideoReader is None or cpu is None:
            raise ImportError("SenseNova InternVL video input requires decord. Install it in the active environment.")
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        frame_indices = self._sample_frame_indices(video_path, vr)
        max_num = max(
            1,
            min(
                self.max_patches,
                int(os.getenv("OOS_INTERNVL_VIDEO_MAX_PATCHES_PER_FRAME", "1")),
            ),
        )
        transform = _build_transform(self.video_height)
        pixel_values_list = []
        num_patches_list = []
        for frame_index in frame_indices:
            img = Image.fromarray(vr[int(frame_index)].asnumpy()).convert("RGB")
            tiles = _dynamic_preprocess(
                img,
                min_num=self.min_patches,
                max_num=max_num,
                image_size=self.video_height,
                use_thumbnail=True,
            )
            pixel_values = torch.stack([transform(tile) for tile in tiles])
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
        return torch.cat(pixel_values_list), num_patches_list

    def _remote_generation_kwargs(self, gen_kwargs: Dict) -> Dict:
        current = dict(gen_kwargs)
        current.pop("until", None)
        current.pop("image_sizes", None)
        current["max_new_tokens"] = int(current.get("max_new_tokens", 1024))
        current["num_beams"] = int(current.get("num_beams", 1))
        temperature = float(current.get("temperature", 0) or 0)
        current["do_sample"] = temperature > 0
        if current["do_sample"]:
            current["temperature"] = temperature
            if current.get("top_p") is not None:
                current["top_p"] = float(current["top_p"])
        else:
            current.pop("temperature", None)
            current.pop("top_p", None)
            current.pop("top_k", None)
        return current

    def _generate_until_remote_chat(self, requests: List[Instance]) -> List[GenerationResult]:
        res: List[GenerationResult] = []

        def _collate(x):
            return x[2], x[2]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0.0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            doc = self.task_dict[task][split][doc_id[0]]
            chat_messages = ChatMessages(**{"messages": doc_to_messages[0](doc)})

            system_text, history, current_text, video_paths = self._remote_chat_parts(chat_messages)
            if system_text:
                self.model.system_message = system_text

            if not video_paths:
                pixel_values = None
                num_patches_list = []
                question = current_text or ctx[0]
            else:
                if len(video_paths) != 1:
                    raise ValueError(f"SenseNova InternVL remote chat supports one video, got {len(video_paths)}")
                pixel_values, num_patches_list = self._load_remote_video(video_paths[0])
                pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
                video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
                question = video_prefix + (current_text or ctx[0])

            self._dbg("\n" + "=" * 100)
            self._dbg(
                f"[STEP] task={task} split={split} doc_id={doc_id[0]} "
                f"step={doc.get('step')} id={doc.get('id')} mode={doc.get('mode')}"
            )
            self._dbg(f"[SYSTEM] {system_text}")
            self._dbg(f"[HISTORY PAIRS] {len(history)}")
            self._dbg(f"[VIDEO PATHS] {video_paths}")
            self._dbg(f"[CURRENT USER TYPES] {['video'] * len(video_paths) + ['text']}")
            self._dbg(f"[QUESTION] {current_text}")
            self._dbg(f"[VIDEO FRAMES] {len(num_patches_list)}")

            gen_kwargs = self._remote_generation_kwargs(all_gen_kwargs[0])
            self._dbg(f"[GEN KWARGS] {gen_kwargs}")
            start_time = time.time()
            answer = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                gen_kwargs,
                num_patches_list=num_patches_list,
                history=history,
                return_history=False,
            )
            end_time = time.time()

            total_elapsed_time += end_time - start_time
            token_counts = None
            if isinstance(answer, str):
                output_tokens = len(self.tokenizer.encode(answer, add_special_tokens=False))
                total_tokens += output_tokens
                token_counts = TokenCounts(output_tokens=output_tokens)

            print("\n" + "=" * 80)
            print("[RAW OUTPUT]")
            print(answer)
            print("\n[CLEANED OUTPUT]")
            print(answer)

            res.append(GenerationResult(text=answer, token_counts=token_counts))
            self.cache_hook.add_partial("generate_until", (question, gen_kwargs), answer)
            pbar.update(1)

        pbar.close()
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0,
            additional_metrics={"rank": self.rank},
        )
        return re_ords.get_original(res)

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        """Generate responses for a list of requests.

        Args:
            requests: List of Instance objects containing generation requests.

        Returns:
            List of generated response strings.
        """
        if self.use_remote_chat:
            return self._generate_until_remote_chat(requests)

        res: List[GenerationResult] = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[2], x[2]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0
        for chunk in chunks:
            self._dbg(f"\n[BATCH] size={len(chunk)}")
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            chat_messages = [doc_to_messages[0](self.task_dict[task][split][ids]) for ids in doc_id]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for idx, messages in enumerate(chat_messages):
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
                request_doc_id = doc_id[idx]
                doc = self.task_dict[task][split][request_doc_id]
                source_name = getattr(doc_to_messages[idx], "__name__", "")
                self._dbg("\n" + "=" * 100)
                self._dbg(
                    f"[STEP] task={task} split={split} doc_id={request_doc_id} "
                    f"step={doc.get('step')} id={doc.get('id')} mode={doc.get('mode')}"
                )
                self._dbg(f"[QUESTION] {doc.get('question')}")
                self._dbg(f"[SOURCE] doc_to_messages={source_name}")
                self._dbg(
                    f"[MEDIA EXTRACTED] images={len(visual)} videos={len(video)} "
                    f"image_urls={visual} video_urls={video}"
                )
                self._dbg("[FINAL PROTOCOL MESSAGES]")
                for msg_idx, msg in enumerate(messages.model_dump()["messages"]):
                    content = msg.get("content", [])
                    self._dbg(
                        f"  [PROTO] idx={msg_idx} role={msg.get('role')} "
                        f"types={[c.get('type') for c in content]} "
                        f"{self._content_preview(content)}"
                    )
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            self._dbg(f"[MEDIA FLATTENED] images={len(visuals)} videos={len(videos)}")

            images_kwargs = {}
            videos_kwargs = {"do_sample_frames": self.do_sample_frames}
            if self.do_resize_video:
                videos_kwargs["do_resize"] = True
                videos_kwargs["size"] = {"height": self.video_height, "width": self.video_width}
            else:
                videos_kwargs["do_resize"] = False
            if self.min_patches is not None:
                images_kwargs["min_patches"] = self.min_patches
            if self.max_patches is not None:
                images_kwargs["max_patches"] = self.max_patches
            if self.do_sample_frames:
                if self.num_frames is not None:
                    videos_kwargs["num_frames"] = self.num_frames
                elif self.fps is not None:
                    videos_kwargs["fps"] = self.fps

            # Apply chat template
            messages = chat_messages[0].model_dump()["messages"]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            self._dbg(f"[CHAT TEMPLATE LEN] {len(text)} chars")
            self._dbg(f"[CHAT TEMPLATE]\n{text}")
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            images = visuals if len(visuals) > 0 else None
            if len(videos) == 0:
                videos = None
            self._dbg(
                f"[PROCESSOR INPUT] images={'None' if images is None else len(images)} "
                f"videos={'None' if videos is None else len(videos)} "
                f"images_kwargs={images_kwargs} videos_kwargs={videos_kwargs}"
            )
            inputs = self.processor(
                images=images,
                videos=videos,
                text=text,
                return_tensors="pt",
                **images_kwargs,
                **videos_kwargs,
            ).to(self.device, self.model.dtype)
            pixel_shape = tuple(inputs.pixel_values.shape) if hasattr(inputs, "pixel_values") else "NA"
            self._dbg(
                f"[PROCESSOR OUTPUT] input_ids_shape={tuple(inputs.input_ids.shape)} "
                f"attention_mask_shape={tuple(inputs.attention_mask.shape) if hasattr(inputs, 'attention_mask') else 'NA'} "
                f"pixel_values_shape={pixel_shape}"
            )

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = dict(all_gen_kwargs[0])
            gen_kwargs.pop("until", None)
            gen_kwargs.pop("image_sizes", None)
            gen_kwargs.pop("do_sample", None)
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            do_sample = True if gen_kwargs["temperature"] > 0 else False
            self._dbg(f"[GEN KWARGS] {gen_kwargs}")
            generated_ids_trimmed = None
            answers = [""]
            try:
                start_time = time.time()
                cont = self.model.generate(
                    **inputs,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eot_token_id,
                    do_sample=do_sample,
                    temperature=gen_kwargs["temperature"] if do_sample else None,
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                end_time = time.time()

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                # Calculate timing metrics
                total_elapsed_time += end_time - start_time
                output_tokens = sum(len(ids) for ids in generated_ids_trimmed)
                total_tokens += output_tokens
                self._dbg(
                    f"[GENERATION METRICS] elapsed={end_time - start_time:.4f}s "
                    f"output_tokens={output_tokens}"
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                self._dbg(f"[GENERATION ERROR] {type(e).__name__}: {e}")
                total_elapsed_time += 0
                total_tokens += 0

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{answers}\n")

            for i, answer in enumerate(answers):
                self._dbg("\n" + "=" * 80)
                self._dbg("[RAW OUTPUT]")
                self._dbg(str(answer))
                print("\n" + "=" * 80)
                print("[RAW OUTPUT]")
                print(answer)
                print("\n[CLEANED OUTPUT]")
                print(answer)
                token_counts = TokenCounts(output_tokens=len(generated_ids_trimmed[i])) if generated_ids_trimmed is not None else None
                res.append(GenerationResult(text=answer, token_counts=token_counts))
                self.cache_hook.add_partial("generate_until", (text, gen_kwargs), answer)
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        metric_dict = {
            "total_gen_tokens": total_tokens,
            "total_elapsed_time": total_elapsed_time,
            "avg_speed": total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood for requests. Not implemented for InternVLHf."""
        # TODO: Implement log-likelihood computation for InternVLHf.
        raise NotImplementedError("Loglikelihood is not implemented for InternVLHf.")

    def generate_until_multi_round(self, requests) -> List[str]:
        return self.generate_until(requests)

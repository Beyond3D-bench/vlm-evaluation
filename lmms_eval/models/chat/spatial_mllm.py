import os
import sys
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2_5_VLProcessor

try:
    import decord
except ImportError:
    decord = None

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


def _prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs, image_temporal_patch_size=2):
    video_tchw = []
    image_tchw = []

    image_temporal_patch_size = int(image_temporal_patch_size)
    if image_temporal_patch_size <= 0:
        raise ValueError(
            "Spatial-MLLM image temporal patch size must be positive, "
            f"got {image_temporal_patch_size}"
        )

    if video_inputs:
        for video_input in video_inputs:
            if isinstance(video_input, torch.Tensor):
                video_input = video_input.float() / 255.0
            elif isinstance(video_input, list) and all(isinstance(img, Image.Image) for img in video_input):
                video_input = torch.stack([torch.tensor(np.array(img)).permute(2, 0, 1) for img in video_input]).float() / 255.0
            else:
                raise ValueError(f"Unsupported Spatial-MLLM video input format: {type(video_input)}")
            if video_input.ndim != 4:
                raise ValueError(
                    "Spatial-MLLM video tensors must have shape (T, C, H, W), "
                    f"got {tuple(video_input.shape)}"
                )
            video_tchw.append(video_input)

    if image_inputs:
        for image_input in image_inputs:
            if isinstance(image_input, Image.Image):
                image_input = torch.tensor(np.array(image_input)).permute(2, 0, 1).unsqueeze(0)
                image_input = image_input.repeat(image_temporal_patch_size, 1, 1, 1).float() / 255.0
            else:
                raise ValueError(f"Unsupported Spatial-MLLM image input format: {type(image_input)}")
            if image_input.ndim != 4:
                raise ValueError(
                    "Spatial-MLLM image tensors must have shape (T, C, H, W), "
                    f"got {tuple(image_input.shape)}"
                )
            image_tchw.append(image_input)

    batch.update(
        {
            "video_tchw": video_tchw if video_tchw else None,
            "image_tchw": image_tchw if image_tchw else None,
        }
    )
    return batch


def _keep_last_token_for_lm_head(_module, args):
    """Avoid materializing prompt-position logits during autoregressive generation."""
    if not args:
        return None

    hidden_states = args[0]
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim < 3 or hidden_states.shape[-2] <= 1:
        return None

    return (hidden_states[..., -1:, :], *args[1:])


@register_model("spatial_mllm_chat")
class SpatialMLLM(lmms):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "Diankun/Spatial-MLLM-v1.1-Instruct-820K",
        spatial_mllm_repo: Optional[str] = None,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 16,
        fps: Optional[float] = None,
        video_resolution: Optional[int] = 448,
        use_fast: bool = False,
        last_token_logits_only: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")
        if video_resolution is not None and (video_resolution <= 0 or video_resolution % 28 != 0):
            raise ValueError(f"video_resolution must be a positive multiple of 28, got {video_resolution}")

        repo_path = spatial_mllm_repo or os.environ.get("SPATIAL_MLLM_REPO")
        if repo_path:
            external_vggt_path = os.path.join(repo_path, "src", "qwenvl", "external")
            for path in (repo_path, external_vggt_path):
                if path not in sys.path:
                    sys.path.append(path)

        try:
            from src.qwenvl.model.spatial_mllm import SpatialMLLMConfig, SpatialMLLMForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Spatial-MLLM source is required. Install/clone https://github.com/THU-SI/Spatial-MLLM "
                "and pass `spatial_mllm_repo=/path/to/Spatial-MLLM` or set SPATIAL_MLLM_REPO."
            ) from exc

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "config": SpatialMLLMConfig.from_pretrained(pretrained),
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = SpatialMLLMForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor = Qwen2_5_VLProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, use_fast=use_fast)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.video_resolution = video_resolution
        self.last_token_logits_only = last_token_logits_only
        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Spatial-MLLM")

    def _log_samples_enabled(self):
        return os.getenv("OOS_SPATIAL_LOG_SAMPLES", os.getenv("OOS_CHAT_DEBUG", "0")) == "1"

    def _sample_log_max_chars(self):
        try:
            return int(os.getenv("OOS_SAMPLE_LOG_MAX_CHARS", "4000"))
        except ValueError:
            return 4000

    def _truncate_for_log(self, text):
        max_chars = self._sample_log_max_chars()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}\n... [truncated {len(text) - max_chars} chars]"

    def _chat_text_for_log(self, chat_message: ChatMessages):
        blocks = []
        for message in chat_message.messages:
            text_parts = [content.text for content in message.content if content.type == "text"]
            if text_parts:
                blocks.append(f"{message.role}: " + "\n".join(text_parts))
        return self._truncate_for_log("\n\n".join(blocks))

    def _doc_meta_for_log(self, doc):
        if not isinstance(doc, dict):
            return {}
        return {
            "id": doc.get("doc_id", doc.get("id", doc.get("trajectory_id"))),
            "video_id": doc.get("source_video_id", doc.get("video_id")),
            "step": doc.get("step"),
            "question_class": doc.get("step_question_class", doc.get("question_class")),
            "answer": doc.get("target_text", doc.get("answer")),
        }

    def _print_sample_log(self, header, sample_index, doc_id, task, split, doc_meta, text):
        print(f"\n[SpatialMLLM][{header}] sample_index={sample_index} task={task} split={split} doc_id={doc_id}", flush=True)
        for key, value in doc_meta.items():
            if value is not None:
                print(f"[SpatialMLLM][{header}] {key}={value}", flush=True)
        if text:
            print(f"[SpatialMLLM][{header}] text:\n{text}", flush=True)

    def _video_kwargs(self, videos):
        if self.video_resolution is not None:
            video_kwargs = {
                "resized_height": self.video_resolution,
                "resized_width": self.video_resolution,
            }
        else:
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
            video_kwargs["max_frames"] = self.max_num_frames
        elif videos and decord is not None:
            try:
                video_total_frames = len(decord.VideoReader(videos[0]))
                nframes = min(self.max_num_frames, video_total_frames)
                nframes = max(2, (nframes // 2) * 2)
                video_kwargs["nframes"] = nframes
            except Exception as exc:
                eval_logger.warning(f"Failed to probe video {videos[0]}: {exc}, using default nframes")
                video_kwargs["nframes"] = self.max_num_frames
        else:
            video_kwargs["nframes"] = self.max_num_frames
        return video_kwargs

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []

        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0
        sample_log_index = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            docs = [self.task_dict[task][split][ids] for ids, task, split in zip(doc_id, task, split)]
            chat_messages = [doc_to_messages[idx](doc) for idx, doc in enumerate(docs)]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]

            sample_indices = list(range(sample_log_index, sample_log_index + len(chat_messages)))
            sample_log_index += len(chat_messages)
            if self._log_samples_enabled():
                for idx, ids, task_name, split_name, doc, chat_message in zip(sample_indices, doc_id, task, split, docs, chat_messages):
                    self._print_sample_log(
                        "BEGIN",
                        idx,
                        ids,
                        task_name,
                        split_name,
                        self._doc_meta_for_log(doc),
                        self._chat_text_for_log(chat_message),
                    )

            videos = []
            for messages in chat_messages:
                _, video, _ = messages.extract_media()
                videos.extend(video)

            batched_messages = [chat_message.to_hf_messages(video_kwargs=self._video_kwargs(videos)) for chat_message in chat_messages]
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(batched_messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
            inputs = _prepare_spatial_mllm_inputs(
                inputs,
                video_inputs,
                image_inputs,
                image_temporal_patch_size=self.config.vision_config.temporal_patch_size,
            )

            inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)
            if inputs.get("image_tchw") is not None:
                inputs["image_tchw"] = [image_tchw.to(self.model.device) for image_tchw in inputs["image_tchw"]]
            if inputs.get("video_tchw") is not None:
                inputs["video_tchw"] = [video_tchw.to(self.model.device) for video_tchw in inputs["video_tchw"]]

            gen_kwargs = all_gen_kwargs[0]
            current_gen_kwargs = {
                "max_new_tokens": 1024,
                "temperature": 0.1,
                "top_p": 0.001,
                "num_beams": 1,
                **gen_kwargs,
            }
            do_sample = current_gen_kwargs["temperature"] > 0
            temperature = current_gen_kwargs["temperature"] if do_sample else None
            top_p = current_gen_kwargs["top_p"] if do_sample else None

            start_time = time.time()
            lm_head_hook = None
            if self.last_token_logits_only:
                # Spatial-MLLM's custom forward predates Transformers' logits_to_keep
                # support and otherwise projects every prompt token over the full
                # vocabulary. Generation only consumes the final-position logits.
                lm_head_hook = self.model.lm_head.register_forward_pre_hook(_keep_last_token_for_lm_head)
            try:
                with torch.no_grad():
                    cont = self.model.generate(
                        **inputs,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        num_beams=current_gen_kwargs["num_beams"],
                        max_new_tokens=current_gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                    )
            finally:
                if lm_head_hook is not None:
                    lm_head_hook.remove()
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for i, (ans, context) in enumerate(zip(answers, texts)):
                res.append(GenerationResult(text=ans, token_counts=TokenCounts(output_tokens=len(generated_ids_trimmed[i]))))
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                if self._log_samples_enabled():
                    self._print_sample_log(
                        "END",
                        sample_indices[i],
                        doc_id[i],
                        task[i],
                        split[i],
                        self._doc_meta_for_log(docs[i]),
                        f"answer: {ans}\noutput_tokens: {len(generated_ids_trimmed[i])}",
                    )
                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Response: {ans}")
            pbar.update(1)

        res = re_ords.get_original(res)
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0,
            additional_metrics={"rank": self.rank},
        )
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Spatial-MLLM")

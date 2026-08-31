import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.phi4_multimodal import Phi4 as Phi4Simple


@register_model("phi4_multimodal_chat_fixed")
class Phi4ChatFixed(Phi4Simple):
    """
    Fixed Phi-4 multimodal chat wrapper for OOS VideoQA-style multi-turn evaluation.

    Main behavior:
    - Uses doc_to_messages when available.
    - Retrieves video separately from doc_to_visual, same as qwen3_vl_chat_fixed.
    - Injects the video into the first user turn.
    - Preserves multi-turn history.
    - Supports OOS_HISTORY_MODE=gold / none / pred.
    - Builds Phi-4 prompt using:
        <|system|>...<|end|>
        <|user|><|image_1|>...question...<|end|>
        <|assistant|>...
    """

    is_simple = False

    # ---------------------------------------------------------------------
    # Debug helpers
    # ---------------------------------------------------------------------

    def _debug_enabled(self) -> bool:
        return os.getenv("OOS_CHAT_DEBUG", "0") == "1"

    def _dbg(self, msg: str) -> None:
        if self._debug_enabled():
            print(msg, flush=True)

    def _content_preview(self, content: List[Dict], max_text_chars: int = 1000) -> str:
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

    def _init_pred_history(self):
        if not hasattr(self, "_pred_history"):
            self._pred_history = defaultdict(dict)

    # ---------------------------------------------------------------------
    # Prompt/message helpers
    # ---------------------------------------------------------------------

    def get_role_tag(self, role: str) -> str:
        if role == "system":
            return "<|system|>"
        elif role == "user":
            return "<|user|>"
        elif role == "assistant":
            return "<|assistant|>"
        else:
            return f"<|{role}|>"

    def _ensure_content_list(self, content):
        if content is None:
            return []
        if isinstance(content, list):
            return list(content)
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        raise TypeError(f"Unsupported message content type: {type(content)}")

    def _normalize_messages(self, raw_messages, doc_id=None) -> List[Dict[str, Any]]:
        if not isinstance(raw_messages, list) or len(raw_messages) == 0:
            raise ValueError(f"doc_to_messages returned invalid payload for doc_id={doc_id}")

        normalized = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                raise TypeError(f"Each message must be a dict, got {type(msg)}")

            role = msg.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Invalid role {role} for doc_id={doc_id}")

            content = self._ensure_content_list(msg.get("content"))
            normalized.append({"role": role, "content": content})

        return normalized

    def _visuals_to_protocol_content(self, visuals):
        """
        Convert doc_to_visual output into chat-style content blocks.

        Your OOS utils.py returns prefix video paths through oos_doc_to_visual().
        Those paths need to be injected into the chat messages manually.
        """
        out = []
        if visuals is None:
            return out

        for visual in visuals:
            if isinstance(visual, str) and visual.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv", ".webm")
            ):
                out.append({"type": "video", "url": visual})
            else:
                out.append({"type": "image", "url": visual})

        return out

    def _image_from_source(self, src):
        if isinstance(src, Image.Image):
            return src
        if isinstance(src, str):
            return Image.open(src).convert("RGB")
        return src

    def _build_phi_prompt(self, messages: List[Dict[str, Any]]):
        """
        Convert normalized chat messages into Phi-4 multimodal prompt.

        Video is represented as sampled frames:
            <|image_1|><|image_2|>...
        and the corresponding PIL images are passed to AutoProcessor.
        """
        prompt_parts = []
        images = []
        audios = []

        image_counter = 1
        audio_counter = 1

        for msg in messages:
            role = msg.get("role", "user")
            prompt_parts.append(self.get_role_tag(role))

            for content in msg.get("content", []):
                ctype = content.get("type")

                if ctype == "text":
                    prompt_parts.append(str(content.get("text", "")))

                elif ctype == "image":
                    src = content.get("url", None)
                    if src is None:
                        src = content.get("image", None)

                    if src is None:
                        continue

                    prompt_parts.append(f"<|image_{image_counter}|>")
                    images.append(self._image_from_source(src))
                    image_counter += 1

                elif ctype == "video":
                    src = content.get("url", None)
                    if src is None:
                        src = content.get("video", None)

                    if src is None:
                        continue

                    frames = self.load_video(src, self.max_frames_num)
                    for frame in frames:
                        prompt_parts.append(f"<|image_{image_counter}|>")
                        images.append(Image.fromarray(np.uint8(frame)).convert("RGB"))
                        image_counter += 1

                elif ctype == "audio":
                    # Not needed for OOS VideoQA.
                    # Kept as a placeholder so audio messages do not crash.
                    # If you later need audio, implement this using Phi4Simple.default_process logic.
                    src = content.get("url", None) or content.get("audio", None)
                    if src is not None:
                        self._dbg(f"[WARNING] Audio content ignored by fixed Phi wrapper: {src}")

                else:
                    self._dbg(f"[WARNING] Unknown content type ignored: {ctype}")

            prompt_parts.append("<|end|>")

        prompt_parts.append("<|assistant|>")

        text = "".join(prompt_parts)
        images = images if len(images) > 0 else None
        audios = audios if len(audios) > 0 else None

        return text, images, audios

    # ---------------------------------------------------------------------
    # Predicted-history helpers
    # ---------------------------------------------------------------------

    def _format_answer_for_history(self, doc: Dict[str, Any], answer_text: str) -> str:
        """
        Same idea as qwen3_vl_chat_fixed:
        store model predictions in a readable form for dependent future steps.
        """
        answer_text = str(answer_text).strip()
        qclass = str(doc.get("step_question_class", "")).strip().lower()
        obj_name = str(doc.get("object_a_name", "the object")).strip() or "the object"

        choices = doc.get("choices") or []
        if choices:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)]
            m = re.search(
                rf"(?:final answer|answer)?\s*[:\-]?\s*([{letters}])\b",
                answer_text,
                flags=re.I,
            )
            if m:
                idx = ord(m.group(1).upper()) - ord("A")
                if 0 <= idx < len(choices):
                    return str(choices[idx])

            pred = answer_text.strip().upper()
            if len(pred) == 1 and pred in letters:
                idx = ord(pred) - ord("A")
                if 0 <= idx < len(choices):
                    return str(choices[idx])

            return answer_text

        m = re.search(
            r"(<TIME\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+video\s+\d+>)"
            r"\s*;\s*Point=\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)",
            answer_text,
            flags=re.I,
        )

        if m:
            time_token = m.group(1)
            x = m.group(2)
            y = m.group(3)

            if qclass == "oos_step2_last_visible":
                return (
                    f"{obj_name} was last visible at {time_token}, "
                    f"at normalized image coordinates (x={x}, y={y}), where x and y are in [0, 1]."
                )

            if qclass == "oos_step3_last_placement":
                return (
                    f"{obj_name} stopped moving at {time_token}, "
                    f"at normalized image coordinates (x={x}, y={y}), where x and y are in [0, 1]."
                )

            return (
                f"The answer is {time_token}, at normalized image coordinates "
                f"(x={x}, y={y}), where x and y are in [0, 1]."
            )

        return answer_text

    def _make_current_user_msg_for_history(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        question_text = str(doc.get("question", "")).strip()

        choices = doc.get("choices") or []
        if choices:
            choice_lines = "\n".join(
                f"{chr(ord('A') + j)}. {choice}"
                for j, choice in enumerate(choices)
            )
            question_text = f"{question_text}\nOptions:\n{choice_lines}"

        return {
            "role": "user",
            "content": [{"type": "text", "text": question_text}],
        }

    # ---------------------------------------------------------------------
    # Main request conversion
    # ---------------------------------------------------------------------

    def _messages_from_request(self, request_args: Tuple):
        ctx, doc_to_source, gen_kwargs, doc_id, task_name, split_name = request_args
        doc = self.task_dict[task_name][split_name][doc_id]

        source_name = getattr(doc_to_source, "__name__", "")

        self._dbg("\n" + "=" * 100)
        self._dbg(
            f"[STEP] task={task_name} split={split_name} doc_id={doc_id} "
            f"step={doc.get('step')} id={doc.get('id')} mode={doc.get('mode')}"
        )
        self._dbg(f"[QUESTION] {doc.get('question')}")
        self._dbg(f"[SOURCE] doc_to_source={source_name}")

        # -----------------------------------------------------------------
        # Chat path: doc_to_messages
        # -----------------------------------------------------------------
        if source_name == "doc_to_messages" or source_name.endswith("doc_to_messages"):
            messages = self._normalize_messages(doc_to_source(doc), doc_id=doc_id)

            # Predicted-history mode:
            # doc_to_messages has no gold history when OOS_HISTORY_MODE=pred,
            # so we insert prior predicted turns here.
            if os.getenv("OOS_HISTORY_MODE", "gold").strip().lower() == "pred":
                self._init_pred_history()

                traj_id = str(
                    doc.get(
                        "trajectory_id",
                        doc.get("source_video_id", doc.get("id")),
                    )
                )
                deps = [str(x) for x in doc.get("depends_on_steps", [])]

                generated_history = []

                for dep_step in deps:
                    if dep_step not in self._pred_history[traj_id]:
                        continue

                    user_msg, answer_text = self._pred_history[traj_id][dep_step]
                    generated_history.append(user_msg)
                    generated_history.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": str(answer_text)}],
                        }
                    )

                if generated_history:
                    if messages and messages[0]["role"] == "system":
                        messages = [messages[0]] + generated_history + messages[1:]
                    else:
                        messages = generated_history + messages

            self._dbg(f"[MULTI-TURN] message_count_before_video={len(messages)}")
            for i, msg in enumerate(messages):
                self._dbg(
                    f"  [MSG BEFORE] idx={i} role={msg['role']} "
                    f"types={[c.get('type') for c in msg['content']]} "
                    f"{self._content_preview(msg['content'])}"
                )

            # Use doc_to_visual as a fallback for task versions whose
            # doc_to_messages contains text/history only.
            bound_task = getattr(doc_to_source, "__self__", None)
            if bound_task is None or not hasattr(bound_task, "doc_to_visual"):
                raise ValueError(
                    f"Could not access doc_to_visual from bound doc_to_messages for task {task_name}"
                )

            visuals = bound_task.doc_to_visual(doc)
            visual_content = self._visuals_to_protocol_content(visuals)

            self._dbg(f"[VISUALS RAW] {visuals}")
            self._dbg(f"[VISUALS PROTOCOL] {visual_content}")

            # Newer OOS doc_to_messages already embeds the video. Only use
            # doc_to_visual as a fallback when the messages contain no media.
            has_embedded_media = any(
                item.get("type") in {"video", "image"}
                for message in messages
                for item in message.get("content", [])
            )

            if visual_content and not has_embedded_media:
                system_msgs = [m for m in messages if m.get("role") == "system"]
                non_system_msgs = [m for m in messages if m.get("role") != "system"]

                first_user_idx = None
                for i, msg in enumerate(non_system_msgs):
                    if msg.get("role") == "user":
                        first_user_idx = i
                        break

                if first_user_idx is None:
                    non_system_msgs = [
                        {
                            "role": "user",
                            "content": visual_content + [{"type": "text", "text": ctx}],
                        }
                    ]
                else:
                    non_system_msgs[first_user_idx]["content"] = (
                        visual_content + non_system_msgs[first_user_idx]["content"]
                    )

                messages = system_msgs + non_system_msgs

                self._dbg("[VIDEO ATTACH] attached video to first user text turn")
                if first_user_idx is not None:
                    self._dbg(
                        f"[VIDEO ATTACH] first_user_types="
                        f"{[c.get('type') for c in non_system_msgs[first_user_idx]['content']]}"
                    )
            elif has_embedded_media:
                self._dbg("[VIDEO ATTACH] using media already embedded by doc_to_messages")
            else:
                self._dbg("[WARNING] visual_content is empty; no video/image attached.")

        # -----------------------------------------------------------------
        # Fallback path: doc_to_visual/doc_to_text-style request
        # -----------------------------------------------------------------
        else:
            visuals = doc_to_source(doc)
            content = self._visuals_to_protocol_content(visuals)
            content.append({"type": "text", "text": ctx})
            messages = [{"role": "user", "content": content}]
            self._dbg("[SINGLE-TURN] fallback path used")

        # Ensure a system message exists.
        has_system = any(msg.get("role") == "system" for msg in messages)
        if not has_system:
            system_prompt = getattr(self, "system_prompt", "You are a helpful assistant.")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            ] + messages
            self._dbg("[SYSTEM] Added wrapper system prompt.")

        self._dbg("[FINAL PROTOCOL MESSAGES]")
        for i, msg in enumerate(messages):
            self._dbg(
                f"  [PROTO] idx={i} role={msg['role']} "
                f"types={[c.get('type') for c in msg['content']]} "
                f"{self._content_preview(msg['content'])}"
            )

        prompt_text, images, audios = self._build_phi_prompt(messages)

        self._dbg(f"[PHI PROMPT LEN] {len(prompt_text)} chars")
        self._dbg(f"[PHI IMAGES] {0 if images is None else len(images)}")

        return prompt_text, images, audios, gen_kwargs

    def _prepare_batch(self, chunk: List[Tuple]):
        """
        Phi-4 multimodal HF processor can handle batching, but for OOS video
        it is safer to use batch_size=1. This method still supports batches
        with left padding through the processor if needed.
        """
        texts = []
        images_batch = []
        audios_batch = []
        gen_kwargs = None

        self._dbg(f"\n[BATCH] size={len(chunk)}")

        for request_args in chunk:
            text, images, audios, request_gen_kwargs = self._messages_from_request(request_args)
            texts.append(text)
            images_batch.append(images)
            audios_batch.append(audios)

            if gen_kwargs is None:
                gen_kwargs = request_gen_kwargs

        # For Phi processor, if batch_size=1, pass the single media list directly.
        # For batch_size>1, this may work depending on processor support, but
        # OOS evaluation is recommended with --batch_size 1.
        if len(chunk) == 1:
            images = images_batch[0]
            audios = audios_batch[0]
            text_arg = texts[0]
        else:
            images = images_batch
            audios = audios_batch
            text_arg = texts

        inputs = self._processor(
            text=text_arg,
            images=images,
            audios=audios,
            return_tensors="pt",
            padding=True if len(chunk) > 1 else False,
        )

        self._dbg(
            f"[PROCESSOR OUTPUT] input_ids_shape={tuple(inputs['input_ids'].shape)}"
        )

        return inputs, texts, gen_kwargs

    # ---------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------

    def _build_generate_kwargs(self, gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        current = dict(gen_kwargs or {})

        current.setdefault("max_new_tokens", 1024)
        current.setdefault("temperature", 0)
        current.setdefault("top_p", None)
        current.setdefault("num_beams", 1)

        temperature = current.get("temperature", 0)

        generate_kwargs = {
            "max_new_tokens": current["max_new_tokens"],
            "use_cache": self.use_cache,
            "pad_token_id": self.eot_token_id,
            "num_beams": current["num_beams"],
            "num_logits_to_keep": 0,
        }

        if temperature and temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
            if current.get("top_p") is not None:
                generate_kwargs["top_p"] = current["top_p"]
        else:
            generate_kwargs["do_sample"] = False

        return generate_kwargs

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res: List[GenerationResult] = []

        pred_mode = os.getenv("OOS_HISTORY_MODE", "gold").strip().lower() == "pred"

        if pred_mode:
            # In pred-history mode, order matters because later steps depend on earlier predictions.
            request_args_list = [reg.args for reg in requests]
            chunks = [
                request_args_list[i : i + self.batch_size]
                for i in range(0, len(request_args_list), self.batch_size)
            ]
            re_ords = None
        else:
            def _collate(x):
                return x[0], x[0]

            re_ords = utils.Collator(
                [reg.args for reg in requests],
                _collate,
                group_fn=lambda x: x[2],
                grouping=True,
            )
            chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))

        pbar = tqdm(
            total=len(chunks),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        total_elapsed_time = 0.0
        total_tokens = 0

        for chunk in chunks:
            inputs, prompt_texts, gen_kwargs = self._prepare_batch(chunk)

            inputs = inputs.to(self.device)

            generate_kwargs = self._build_generate_kwargs(gen_kwargs)

            start_time = time.time()
            try:
                cont = self.model.generate(
                    **inputs,
                    **generate_kwargs,
                )
            except Exception as e:
                eval_logger.exception(f"Error generating text: {e}")
                cont = inputs["input_ids"]
            end_time = time.time()

            input_len = inputs["input_ids"].shape[-1]
            generated_ids_trimmed = cont[:, input_len:]

            answers = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
            )

            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for i, (ans, prompt_text) in enumerate(zip(answers, prompt_texts)):
                cleaned_ans = str(ans).strip()

                if os.getenv("OOS_HISTORY_MODE", "gold").strip().lower() == "pred":
                    self._init_pred_history()

                    request_args = chunk[i]
                    _, doc_to_source, _, doc_id, task_name, split_name = request_args
                    doc = self.task_dict[task_name][split_name][doc_id]

                    traj_id = str(
                        doc.get(
                            "trajectory_id",
                            doc.get("source_video_id", doc.get("id")),
                        )
                    )
                    step_id = str(doc.get("step"))

                    user_msg = self._make_current_user_msg_for_history(doc)
                    history_answer = self._format_answer_for_history(doc, cleaned_ans)

                    self._pred_history[traj_id][step_id] = (user_msg, history_answer)

                if self._debug_enabled():
                    print("\n" + "=" * 80)
                    print("[RAW OUTPUT]")
                    print(ans)
                    print("\n[CLEANED OUTPUT]")
                    print(cleaned_ans)

                output_token_count = len(generated_ids_trimmed[i])

                res.append(
                    GenerationResult(
                        text=cleaned_ans,
                        token_counts=TokenCounts(output_tokens=output_token_count),
                    )
                )

                self.cache_hook.add_partial(
                    "generate_until",
                    (prompt_text, gen_kwargs),
                    cleaned_ans,
                )

                eval_logger.debug(f"Prompt: {prompt_text}")
                eval_logger.debug(f"Model Response: {cleaned_ans}")

            pbar.update(1)

            # Optional cleanup for long videos.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if re_ords is not None:
            res = re_ords.get_original(res)

        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0.0
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=avg_speed,
            additional_metrics={"rank": self.rank},
        )

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[GenerationResult]:
        return self.generate_until(requests)

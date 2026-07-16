import os
import time
from typing import Dict, List, Tuple
from collections import defaultdict

from loguru import logger as eval_logger
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import gc
import torch
import re

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3_VLSimple
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_chat_fixed")
class Qwen3_VL_Chat_Fixed(Qwen3_VLSimple):
    is_simple = False

    def _debug_enabled(self) -> bool:
        return os.getenv("OOS_CHAT_DEBUG", "0") == "1"

    def _dbg(self, msg: str) -> None:
        if self._debug_enabled():
            print(msg, flush=True)

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

    def _init_pred_history(self):
        if not hasattr(self, "_pred_history"):
            self._pred_history = defaultdict(dict)

    def _messages_from_request(
        self,
        request_args: Tuple,
    ) -> Tuple[List[Dict], Dict, str]:
        ctx, doc_to_source, gen_kwargs, doc_id, task_name, split_name = request_args
        task_container = self.task_dict[task_name]
        doc = task_container[split_name][doc_id]

        def _ensure_content_list(content):
            if content is None:
                return []
            if isinstance(content, list):
                return list(content)
            if isinstance(content, str):
                return [{"type": "text", "text": content}]
            raise TypeError(f"Unsupported message content type: {type(content)}")

        def _normalize_messages(raw_messages):
            if not isinstance(raw_messages, list) or len(raw_messages) == 0:
                raise ValueError(f"doc_to_messages returned invalid payload for doc_id={doc_id}")

            normalized = []
            for msg in raw_messages:
                if not isinstance(msg, dict):
                    raise TypeError(f"Each message must be a dict, got {type(msg)}")
                role = msg.get("role")
                if role not in {"system", "user", "assistant"}:
                    raise ValueError(f"Invalid role {role} for doc_id={doc_id}")

                content = _ensure_content_list(msg.get("content"))
                normalized.append({"role": role, "content": content})
            return normalized

        def _visuals_to_protocol_content(visuals):
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

        def _messages_have_media(messages):
            for msg in messages:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                if any(item.get("type") in {"image", "video"} for item in content if isinstance(item, dict)):
                    return True
            return False

        source_name = getattr(doc_to_source, "__name__", "")

        self._dbg("\n" + "=" * 100)
        self._dbg(
            f"[STEP] task={task_name} split={split_name} doc_id={doc_id} "
            f"step={doc.get('step')} id={doc.get('id')} mode={doc.get('mode')}"
        )
        self._dbg(f"[QUESTION] {doc.get('question')}")
        self._dbg(f"[SOURCE] doc_to_source={source_name}")

        if "messages" in source_name:
            messages = _normalize_messages(doc_to_source(doc))
            if os.getenv("OOS_HISTORY_MODE", "gold").strip().lower() == "pred":
                self._init_pred_history()

                traj_id = str(doc.get("trajectory_id", doc.get("source_video_id", doc.get("id"))))
                deps = [str(x) for x in doc.get("depends_on_steps", [])]

                generated_history = []

                for dep_step in deps:
                    if dep_step not in self._pred_history[traj_id]:
                        continue

                    user_msg, a_text = self._pred_history[traj_id][dep_step]

                    generated_history.append(user_msg)
                    generated_history.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": str(a_text)}],
                    })

                # insert after system, before current user
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

            # Sanity check for intended history behavior:
            # step 1 should have only system + current user question before video injection
            step_no = doc.get("step")
            if isinstance(step_no, int):
                non_system_before = [m for m in messages if m["role"] != "system"]
                self._dbg(f"[CHECK] non_system_messages_before_video={len(non_system_before)}")
                if step_no == 1:
                    self._dbg(
                        "[CHECK] step=1 expected: no previous gold QA, only current question "
                        "(plus optional system message)."
                    )
                else:
                    self._dbg(
                        f"[CHECK] step={step_no} expected: only previous QA history + current question."
                    )

            has_embedded_media = _messages_have_media(messages)
            if has_embedded_media:
                visuals = []
                visual_content = []
                self._dbg("[VISUALS EMBEDDED] using media from doc_to_messages")
            else:
                bound_task = getattr(doc_to_source, "__self__", None)
                if bound_task is None or not hasattr(bound_task, "doc_to_visual"):
                    raise ValueError(
                        f"Could not access doc_to_visual from bound doc_to_messages for task {task_name}"
                    )

                visuals = bound_task.doc_to_visual(doc)
                visual_content = _visuals_to_protocol_content(visuals)

            self._dbg(f"[VISUALS RAW] {visuals}")
            self._dbg(f"[VISUALS PROTOCOL] {visual_content}")

            # # system prompt + previous history (if any) + videos + current question
            # if visual_content:
            #     last_user_idx = None
            #     for i in range(len(messages) - 1, -1, -1):
            #         if messages[i]["role"] == "user":
            #             last_user_idx = i
            #             break

            #     if last_user_idx is None:
            #         messages.append(
            #             {"role": "user", "content": visual_content + [{"type": "text", "text": ctx}]}
            #         )
            #         last_user_idx = len(messages) - 1
            #     else:
            #         messages[last_user_idx]["content"] = (
            #             visual_content + messages[last_user_idx]["content"]
            #         )

            #     self._dbg(f"[VIDEO ATTACH] attached_to_user_idx={last_user_idx}")
            #     self._dbg(
            #         f"[VIDEO ATTACH] final_user_types="
            #         f"{[c.get('type') for c in messages[last_user_idx]['content']]}"
            #     )

            #     attached_videos = [
            #         c.get("url") for c in messages[last_user_idx]["content"] if c.get("type") == "video"
            #     ]
            #     self._dbg(f"[VIDEO ATTACH] attached_video_urls={attached_videos}")

            #     if not attached_videos:
            #         self._dbg("[WARNING] No video attached to final user turn.")
            # else:
            #     self._dbg("[WARNING] visual_content is empty; no video/image attached for this step.")

            # video + flattened previous hostory + current step question
            # if visual_content:
            #     # Flatten previous chat history and current question into one final user turn:
            #     # user content = [video] + [previous-step text + current-step text]

            #     system_msgs = [m for m in messages if m.get("role") == "system"]
            #     non_system_msgs = [m for m in messages if m.get("role") != "system"]

            #     def _text_from_content(content):
            #         return "\n".join(
            #             str(part.get("text", ""))
            #             for part in content
            #             if part.get("type") == "text" and str(part.get("text", "")).strip()
            #         ).strip()

            #     # The last user message should be the current step question.
            #     last_user_idx = None
            #     for i in range(len(non_system_msgs) - 1, -1, -1):
            #         if non_system_msgs[i].get("role") == "user":
            #             last_user_idx = i
            #             break

            #     if last_user_idx is None:
            #         current_text = ctx
            #         history_msgs = non_system_msgs
            #     else:
            #         current_text = _text_from_content(non_system_msgs[last_user_idx].get("content", []))
            #         history_msgs = non_system_msgs[:last_user_idx]

            #     history_lines = []
            #     pair_no = 1
            #     pending_question = None

            #     for msg in history_msgs:
            #         role = msg.get("role")
            #         text = _text_from_content(msg.get("content", []))
            #         if not text:
            #             continue

            #         if role == "user":
            #             pending_question = text
            #         elif role == "assistant":
            #             if pending_question is not None:
            #                 history_lines.append(f"Previous step {pair_no} question:\n{pending_question}")
            #                 history_lines.append(f"Previous step {pair_no} answer:\n{text}")
            #                 pair_no += 1
            #                 pending_question = None
            #             else:
            #                 history_lines.append(f"Previous assistant answer:\n{text}")

            #     # If there is an unmatched previous user message, keep it as context.
            #     if pending_question is not None:
            #         history_lines.append(f"Previous step question:\n{pending_question}")

            #     merged_text_parts = []
            #     if history_lines:
            #         merged_text_parts.append("Previous steps:")
            #         merged_text_parts.append("\n\n".join(history_lines))

            #     merged_text_parts.append("Current step:")
            #     merged_text_parts.append(current_text or ctx)

            #     merged_text = "\n\n".join(merged_text_parts)

            #     messages = system_msgs + [
            #         {
            #             "role": "user",
            #             "content": visual_content + [{"type": "text", "text": merged_text}],
            #         }
            #     ]

            #     self._dbg("[VIDEO ATTACH OPTION 1] flattened into one user turn")
            #     self._dbg(
            #         f"[VIDEO ATTACH OPTION 1] final_user_types="
            #         f"{[c.get('type') for c in messages[-1]['content']]}"
            #     )
            # else:
            #     self._dbg("[WARNING] visual_content is empty; no video/image attached for this step.")

            # video + previous history (chat turns) + current step question
            if visual_content:
                system_msgs = [m for m in messages if m.get("role") == "system"]
                non_system_msgs = [m for m in messages if m.get("role") != "system"]

                current_user_idx = None
                for i in range(len(non_system_msgs) - 1, -1, -1):
                    msg = non_system_msgs[i]
                    if msg.get("role") == "user":
                        current_user_idx = i
                        break

                if current_user_idx is None:
                    non_system_msgs = [{
                        "role": "user",
                        "content": visual_content + [{"type": "text", "text": ctx}],
                    }]
                    current_user_idx = 0
                else:
                    non_system_msgs[current_user_idx]["content"] = (
                        visual_content + non_system_msgs[current_user_idx]["content"]
                    )

                messages = system_msgs + non_system_msgs

                self._dbg("[VIDEO ATTACH] attached video to current user text turn")
                self._dbg(
                    f"[VIDEO ATTACH] current_user_types="
                    f"{[c.get('type') for c in non_system_msgs[current_user_idx]['content']]}"
                )
            elif not has_embedded_media:
                self._dbg("[WARNING] visual_content is empty; no video/image attached for this step.")

        else:
            visuals = doc_to_source(doc)
            content: List[Dict] = _visuals_to_protocol_content(visuals)
            content.append({"type": "text", "text": ctx})
            messages = [{"role": "user", "content": content}]
            self._dbg("[SINGLE-TURN] fallback path used")
            self._dbg(f"[VISUALS RAW] {visuals}")
            self._dbg(f"[SINGLE-TURN] content={self._content_preview(content)}")

        has_system = any(msg.get("role") == "system" for msg in messages)
        if not has_system:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            ] + messages
            self._dbg("[SYSTEM] Added wrapper system prompt because task messages had no system role.")

        self._dbg("[FINAL PROTOCOL MESSAGES]")
        for i, msg in enumerate(messages):
            self._dbg(
                f"  [PROTO] idx={i} role={msg['role']} "
                f"types={[c.get('type') for c in msg['content']]} "
                f"{self._content_preview(msg['content'])}"
            )

        chat_message = ChatMessages(messages=messages)
        hf_messages = chat_message.to_hf_messages(video_kwargs=self._build_video_kwargs())

        self._dbg("[FINAL HF MESSAGES]")
        for i, msg in enumerate(hf_messages):
            self._dbg(
                f"  [HF] idx={i} role={msg['role']} "
                f"types={[c.get('type') for c in msg['content']]}"
            )
            for item in msg["content"]:
                if item.get("type") == "video":
                    self._dbg(f"    [HF VIDEO] {item.get('video')}")
                elif item.get("type") == "text":
                    text = str(item.get("text", "")).replace("\n", " ").strip()
                    if len(text) > 2000:
                        text = text[:2000] + "..."
                    self._dbg(f"    [HF TEXT] {text}")

        text_prompt = self._apply_chat_template([hf_messages])[0]
        self._dbg(f"[CHAT TEMPLATE LEN] {len(text_prompt)} chars")

        return hf_messages, gen_kwargs, text_prompt

    def _prepare_batch(self, chunk: List[Tuple]):
        batched_messages = []
        prompt_texts = []
        gen_kwargs = None

        self._dbg(f"\n[BATCH] size={len(chunk)}")

        for request_args in chunk:
            hf_messages, request_gen_kwargs, text_prompt = self._messages_from_request(request_args)
            batched_messages.append(hf_messages)
            prompt_texts.append(text_prompt)
            if gen_kwargs is None:
                gen_kwargs = request_gen_kwargs

        image_inputs, video_inputs, processed_video_kwargs = process_vision_info(
            batched_messages,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )

        self._dbg(
            f"[VISION INFO] image_inputs={'None' if image_inputs is None else len(image_inputs)} "
            f"video_inputs={'None' if video_inputs is None else len(video_inputs)} "
            f"processed_video_kwargs={processed_video_kwargs}"
        )

        video_metadata_list = None
        if video_inputs is not None:
            video_inputs, video_metadata_list = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadata_list = list(video_metadata_list)

            self._dbg(f"[VIDEO METADATA COUNT] {len(video_metadata_list)}")
            for i, meta in enumerate(video_metadata_list):
                if isinstance(meta, dict):
                    self._dbg(
                        f"  [VIDEO META {i}] fps={meta.get('fps')} "
                        f"num_frames={len(meta.get('frames_indices', [])) if meta.get('frames_indices') is not None else 'NA'}"
                    )

        texts = self._apply_chat_template(batched_messages)
        self._dbg(f"[TOKENIZATION] num_texts={len(texts)}")

        if self.batch_size > 1:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                return_tensors="pt",
            )

        self._dbg(
            f"[PROCESSOR OUTPUT] input_ids_shape={tuple(inputs.input_ids.shape)} "
            f"attention_mask_shape={tuple(inputs.attention_mask.shape) if hasattr(inputs, 'attention_mask') else 'NA'}"
        )

        return inputs, texts, gen_kwargs

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res: List[GenerationResult] = []

        pred_mode = os.getenv("OOS_HISTORY_MODE", "gold").strip().lower() == "pred"

        if pred_mode:
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
        num_iters = len(chunks)
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        total_elapsed_time = 0.0
        total_tokens = 0

        for chunk in chunks:
            inputs, texts, gen_kwargs = self._prepare_batch(chunk)

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            generate_kwargs = self._build_generate_kwargs(gen_kwargs)

            start_time = time.time()
            cont = self.model.generate(**inputs, **generate_kwargs)
            end_time = time.time()

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for i, (ans, context) in enumerate(zip(answers, texts)):
                raw_ans = ans  # before cleaning
                cleaned_ans = self._strip_thinking(ans)

                # Store model-generated answer for later dependent steps
                if os.getenv("OOS_HISTORY_MODE", "gold").strip().lower() == "pred":
                    self._init_pred_history()

                    request_args = chunk[i]
                    _, doc_to_source, _, doc_id, task_name, split_name = request_args
                    doc = self.task_dict[task_name][split_name][doc_id]

                    traj_id = str(doc.get("trajectory_id", doc.get("source_video_id", doc.get("id"))))
                    step_id = str(doc.get("step"))
                    source_name = getattr(doc_to_source, "__name__", "")

                    # if "doc_to_messages" in source_name:
                    #     msgs = doc_to_source(doc)
                    #     user_msg = msgs[-1]
                    # else:
                    #     user_msg = {
                    #         "role": "user",
                    #         "content": [{"type": "text", "text": doc.get("question", "")}],
                    #     }

                    # self._pred_history[traj_id][step_id] = (user_msg, cleaned_ans)       
                    question_text = str(doc.get("question", "")).strip()

                    choices = doc.get("choices") or []
                    if choices:
                        choice_lines = "\n".join(
                            f"{chr(ord('A') + j)}. {choice}"
                            for j, choice in enumerate(choices)
                        )
                        question_text = f"{question_text}\nOptions:\n{choice_lines}"

                    user_msg = {
                        "role": "user",
                        "content": [{"type": "text", "text": question_text}],
                    }


                    def _format_answer_for_history(doc, answer_text: str) -> str:
                        answer_text = str(answer_text).strip()
                        qclass = str(doc.get("step_question_class", "")).strip().lower()
                        obj_name = str(doc.get("object_a_name", "the object")).strip() or "the object"

                        # Convert multiple-choice letter to semantic choice text.
                        choices = doc.get("choices") or []
                        if choices:
                            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(choices)]
                            m = re.search(
                                rf"(?:final answer|answer)?\s*[:\-]?\s*([{letters}])\b",
                                answer_text,
                                flags=re.I,
                            )
                            if m:
                                idx = ord(m.group(1).upper()) - ord("A")
                                if 0 <= idx < len(choices):
                                    return str(choices[idx])

                            # Also handle exact single-letter output like "B".
                            pred = answer_text.strip().upper()
                            if len(pred) == 1 and pred in letters:
                                idx = ord(pred) - ord("A")
                                if 0 <= idx < len(choices):
                                    return str(choices[idx])

                            return answer_text

                        # Convert structured time/point answer to a clearer sentence.
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


                    history_answer = _format_answer_for_history(doc, cleaned_ans)
                    self._pred_history[traj_id][step_id] = (user_msg, history_answer)

                print("\n" + "=" * 80)
                print(f"[RAW OUTPUT]")
                print(raw_ans)

                print(f"\n[CLEANED OUTPUT]")
                print(cleaned_ans)

                res.append(
                    GenerationResult(
                        text=cleaned_ans,
                        token_counts=TokenCounts(output_tokens=len(generated_ids_trimmed[i])),
                    )
                )

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), cleaned_ans)
                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Response: {ans}")

            pbar.update(1)

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

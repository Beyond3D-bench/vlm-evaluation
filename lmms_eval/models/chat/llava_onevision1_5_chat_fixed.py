import time
from typing import List

import numpy as np
import torch
import re
from loguru import logger as eval_logger
from tqdm import tqdm
import os
from collections import defaultdict

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.llava_onevision1_5 import (
    Llava_OneVision1_5 as LlavaOneVisionSimple,
)
from lmms_eval.protocol import ChatMessages

process_vision_info, _ = optional_import("qwen_vl_utils", "process_vision_info")


def _debug_tensor_dict(name, obj):
    eval_logger.info(f"\n===== {name} =====")

    if hasattr(obj, "items"):
        items = obj.items()
    elif isinstance(obj, dict):
        items = obj.items()
    else:
        eval_logger.info(f"{type(obj)}")
        return

    for k, v in items:
        if torch.is_tensor(v):
            if v.numel() == 0:
                min_val, max_val = "empty", "empty"
            elif v.is_floating_point():
                min_val = v.min().item()
                max_val = v.max().item()
            else:
                min_val = v.min().item()
                max_val = v.max().item()

            nan = torch.isnan(v.float()).any().item() if v.is_floating_point() else False
            inf = torch.isinf(v.float()).any().item() if v.is_floating_point() else False

            eval_logger.info(
                f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                f"device={v.device}, min={min_val}, max={max_val}, "
                f"nan={nan}, inf={inf}"
            )
        else:
            eval_logger.info(f"{k}: type={type(v)}, value={v}")

    eval_logger.info("====================\n")

@register_model("llava_onevision1_5_chat_fixed")
class Llava_OneVision1_5_Chat_Fixed(LlavaOneVisionSimple):
    is_simple = False
    fps = None
    def _debug_enabled(self) -> bool:
        return os.getenv("OOS_CHAT_DEBUG", "1") == "1"

    def _dbg(self, msg: str) -> None:
        if self._debug_enabled():
            print(msg, flush=True)

    def _content_preview(self, content, max_text_chars: int = 2000) -> str:
        parts = []
        if content is None:
            return "None"

        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue

            ctype = item.get("type")
            if ctype == "text":
                text = str(item.get("text", "")).replace("\n", " ").strip()
                if len(text) > max_text_chars:
                    text = text[:max_text_chars] + "..."
                parts.append(f"text='{text}'")
            elif ctype == "video":
                parts.append(f"video='{item.get('video', item.get('url', ''))}'")
            elif ctype == "image":
                parts.append(f"image='{item.get('image', item.get('url', ''))}'")
            else:
                parts.append(str(item))

        return " | ".join(parts)
    def _init_pred_history(self):
        if not hasattr(self, "_pred_history"):
            self._pred_history = defaultdict(dict)

    def _video_attach_mode(self) -> str:
        # "flatten" = Option 1: video + flattened previous history + current question
        # "first_user" = Option 2: video attached to first user turn, keep chat turns
        return os.getenv("OOS_VIDEO_ATTACH_MODE", "flatten").strip().lower()
    
    def _history_user_msg_from_doc(self, doc):
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

    def _format_answer_for_history(self, doc, answer_text: str) -> str:
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

            pred = answer_text.strip().upper()
            if len(pred) == 1 and pred in letters:
                idx = ord(pred) - ord("A")
                if 0 <= idx < len(choices):
                    return str(choices[idx])

            return answer_text

        # Convert structured time/point answer to clearer history text.
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

            coord_text = (
                f"at normalized image coordinates "
                f"(x={x} from the left edge, y={y} from the top edge), "
                f"where x and y are in [0, 1]"
            )

            if qclass == "oos_step2_last_visible":
                return f"{obj_name} was last visible at {time_token}, {coord_text}."

            if qclass == "oos_step3_last_placement":
                return f"{obj_name} stopped moving at {time_token}, {coord_text}."

            return f"The answer is {time_token}, {coord_text}."

        return answer_text

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        assert process_vision_info is not None, "qwen_vl_utils is required. Please install it via `pip install qwen-vl-utils`"

        res = []

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
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        total_elapsed_time = 0.0
        total_tokens = 0

        if self.batch_size > 1:
            self.processor.tokenizer.padding_side = "left"

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]

            raw_messages_list = []

            for ids in doc_id:
                doc = self.task_dict[task][split][ids]
                messages = doc_to_messages[0](doc)

                def _messages_have_media(messages):
                    for msg in messages:
                        content = msg.get("content")
                        if not isinstance(content, list):
                            continue
                        if any(
                            item.get("type") in {"image", "video"}
                            for item in content
                            if isinstance(item, dict)
                        ):
                            return True
                    return False

                if pred_mode:
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

                    if generated_history:
                        if messages and messages[0]["role"] == "system":
                            messages = [messages[0]] + generated_history + messages[1:]
                        else:
                            messages = generated_history + messages

                source_name = getattr(doc_to_messages[0], "__name__", "")

                self._dbg("\n" + "=" * 100)
                self._dbg(
                    f"[STEP] task={task} split={split} doc_id={ids} "
                    f"step={doc.get('step')} id={doc.get('id')} mode={doc.get('mode')}"
                )
                self._dbg(f"[QUESTION] {doc.get('question')}")
                self._dbg(f"[SOURCE] doc_to_messages={source_name}")

                self._dbg(f"[MESSAGES INITIAL] count={len(messages)}")
                for mi, msg in enumerate(messages):
                    self._dbg(
                        f"  [MSG INITIAL] idx={mi} role={msg.get('role')} "
                        f"types={[c.get('type') for c in msg.get('content', [])] if isinstance(msg.get('content'), list) else type(msg.get('content'))} "
                        f"{self._content_preview(msg.get('content'))}"
                    )
                has_embedded_media = _messages_have_media(messages)
                bound_task = getattr(doc_to_messages[0], "__self__", None)
                if has_embedded_media:
                    self._dbg("[VISUALS EMBEDDED] using media from doc_to_messages")
                elif bound_task is not None and hasattr(bound_task, "doc_to_visual"):
                    visuals = bound_task.doc_to_visual(doc)
                    visual_content = []

                    for visual in visuals:
                        if isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                            visual_content.append({"type": "video", "video": visual})
                        else:
                            visual_content.append({"type": "image", "image": visual})

                    self._dbg(f"[VISUALS RAW] {visuals}")
                    self._dbg(f"[VISUALS PROTOCOL] {visual_content}")

                    if visual_content:
                        attach_mode = self._video_attach_mode()

                        system_msgs = [m for m in messages if m.get("role") == "system"]
                        non_system_msgs = [m for m in messages if m.get("role") != "system"]

                        def _ensure_list_content(msg):
                            if not isinstance(msg.get("content"), list):
                                msg["content"] = [
                                    {"type": "text", "text": str(msg.get("content", ""))}
                                ]
                            return msg

                        def _text_from_content(content):
                            if content is None:
                                return ""
                            if isinstance(content, str):
                                return content.strip()
                            return "\n".join(
                                str(part.get("text", ""))
                                for part in content
                                if isinstance(part, dict)
                                and part.get("type") == "text"
                                and str(part.get("text", "")).strip()
                            ).strip()

                        for msg in non_system_msgs:
                            _ensure_list_content(msg)

                        if attach_mode in {"flatten", "option1", "option_1"}:
                            # OPTION 1:
                            # Flatten previous QA and current question into one user turn:
                            # system
                            # user: video + previous steps + current step

                            last_user_idx = None
                            for j in range(len(non_system_msgs) - 1, -1, -1):
                                if non_system_msgs[j].get("role") == "user":
                                    last_user_idx = j
                                    break

                            if last_user_idx is None:
                                current_text = ctx[0] if isinstance(ctx, tuple) else str(ctx)
                                history_msgs = non_system_msgs
                            else:
                                current_text = _text_from_content(
                                    non_system_msgs[last_user_idx].get("content", [])
                                )
                                history_msgs = non_system_msgs[:last_user_idx]

                            history_lines = []
                            pair_no = 1
                            pending_question = None

                            for msg in history_msgs:
                                role = msg.get("role")
                                text = _text_from_content(msg.get("content", []))
                                if not text:
                                    continue

                                if role == "user":
                                    pending_question = text
                                elif role == "assistant":
                                    if pending_question is not None:
                                        history_lines.append(
                                            f"Previous step {pair_no} question:\n{pending_question}"
                                        )
                                        history_lines.append(
                                            f"Previous step {pair_no} answer:\n{text}"
                                        )
                                        pair_no += 1
                                        pending_question = None
                                    else:
                                        history_lines.append(
                                            f"Previous assistant answer:\n{text}"
                                        )

                            if pending_question is not None:
                                history_lines.append(
                                    f"Previous step question:\n{pending_question}"
                                )

                            merged_text_parts = []
                            if history_lines:
                                merged_text_parts.append("Previous steps:")
                                merged_text_parts.append("\n\n".join(history_lines))

                            merged_text_parts.append("Current step:")
                            merged_text_parts.append(current_text)

                            merged_text = "\n\n".join(merged_text_parts)

                            messages = system_msgs + [
                                {
                                    "role": "user",
                                    "content": visual_content + [
                                        {"type": "text", "text": merged_text}
                                    ],
                                }
                            ]

                            self._dbg("[VIDEO ATTACH OPTION 1] flattened into one user turn")
                            self._dbg(
                                f"[VIDEO ATTACH OPTION 1] final_user_types="
                                f"{[c.get('type') for c in messages[-1]['content']]}"
                            )

                        elif attach_mode in {"first_user", "option2", "option_2"}:
                            # OPTION 2:
                            # Keep previous steps as real chat turns, but attach video
                            # to the first user turn:
                            # system
                            # user: video + previous/current first question
                            # assistant: previous answer
                            # user: next/current question

                            first_user_idx = None
                            for j, msg in enumerate(non_system_msgs):
                                if msg.get("role") == "user":
                                    first_user_idx = j
                                    break

                            if first_user_idx is None:
                                attached_idx = 0
                                non_system_msgs = [
                                    {
                                        "role": "user",
                                        "content": visual_content + [
                                            {
                                                "type": "text",
                                                "text": ctx[0] if isinstance(ctx, tuple) else str(ctx),
                                            }
                                        ],
                                    }
                                ]
                            else:
                                attached_idx = first_user_idx
                                non_system_msgs[attached_idx]["content"] = (
                                    visual_content
                                    + non_system_msgs[attached_idx]["content"]
                                )

                            messages = system_msgs + non_system_msgs

                            self._dbg("[VIDEO ATTACH OPTION 2] attached video to first user turn")
                            self._dbg(
                                f"[VIDEO ATTACH OPTION 2] attached_user_idx={attached_idx} "
                                f"types={[c.get('type') for c in non_system_msgs[attached_idx]['content']]}"
                            )

                        else:
                            raise ValueError(
                                f"Unsupported OOS_VIDEO_ATTACH_MODE={attach_mode}. "
                                "Use 'flatten' or 'first_user'."
                            )

                    else:
                        self._dbg("[WARNING] visual_content is empty; no video/image attached for this step.")

                    self._dbg(f"[MULTI-TURN] message_count_after_history={len(messages)}")
                    for mi, msg in enumerate(messages):
                        self._dbg(
                            f"  [MSG AFTER HISTORY] idx={mi} role={msg.get('role')} "
                            f"types={[c.get('type') for c in msg.get('content', [])] if isinstance(msg.get('content'), list) else type(msg.get('content'))} "
                            f"{self._content_preview(msg.get('content'))}"
                        )

                    step_no = doc.get("step")
                    if isinstance(step_no, int):
                        non_system = [m for m in messages if m.get("role") != "system"]
                        self._dbg(f"[CHECK] non_system_messages_after_history={len(non_system)}")
                        if step_no == 1:
                            self._dbg("[CHECK] step=1 expected: no previous generated QA, only current question.")
                        else:
                            self._dbg(f"[CHECK] step={step_no} expected: previous generated QA history + current question.")

                raw_messages_list.append(messages)

            for messages in raw_messages_list:
                for msg in messages:
                    if isinstance(msg.get("content"), list):
                        for item in msg["content"]:
                            if item.get("type") == "video" and "video" in item:
                                item["url"] = item.pop("video")
            chat_messages_list: List[ChatMessages] = [ChatMessages(**{"messages": m}) for m in raw_messages_list]

            self._dbg("[FINAL PROTOCOL MESSAGES]")
            for bi, messages in enumerate(raw_messages_list):
                self._dbg(f"  [BATCH ITEM {bi}]")
                for mi, msg in enumerate(messages):
                    self._dbg(
                        f"    [PROTO] idx={mi} role={msg.get('role')} "
                        f"types={[c.get('type') for c in msg.get('content', [])] if isinstance(msg.get('content'), list) else type(msg.get('content'))} "
                        f"{self._content_preview(msg.get('content'))}"
                    )

            # Prepare video processing kwargs
            video_kwargs = self._build_video_kwargs()

            # Build HF messages and apply chat template
            hf_messages_list = [cm.to_hf_messages(video_kwargs=video_kwargs) for cm in chat_messages_list]

            self._dbg("[FINAL HF MESSAGES]")
            for bi, hf_messages in enumerate(hf_messages_list):
                self._dbg(f"  [BATCH ITEM {bi}]")
                for mi, msg in enumerate(hf_messages):
                    self._dbg(
                        f"    [HF] idx={mi} role={msg.get('role')} "
                        f"types={[c.get('type') for c in msg.get('content', [])]}"
                    )
                    for item in msg.get("content", []):
                        if item.get("type") == "video":
                            self._dbg(f"      [HF VIDEO] {item.get('video', item.get('url'))}")
                        elif item.get("type") == "image":
                            self._dbg(f"      [HF IMAGE] {item.get('image', item.get('url'))}")
                        elif item.get("type") == "text":
                            text = str(item.get("text", "")).replace("\n", " ").strip()
                            if len(text) > 2000:
                                text = text[:2000] + "..."
                            self._dbg(f"      [HF TEXT] {text}")

            texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in hf_messages_list]
            for bi, text in enumerate(texts):
                self._dbg(f"[CHAT TEMPLATE LEN] batch={bi} len={len(text)} chars")

            if self.rank == 0 and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{texts[0]}\n")

            # Extract image/video inputs consistent with OneVision processing
            image_inputs, video_inputs, processed_video_kwargs = process_vision_info(
                hf_messages_list,
                return_video_kwargs=True,
                image_patch_size=14,
                return_video_metadata=True,
            )
            if image_inputs is not None and len(image_inputs) == 0:
                image_inputs = None

            if video_inputs is not None and len(video_inputs) == 0:
                video_inputs = None

            video_metadata_list = None
            if video_inputs is not None:
                video_inputs, video_metadata_list = map(list, zip(*video_inputs))
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                padding=True,
                return_tensors="pt",
            )
            if self.rank == 0:
                eval_logger.info(f"doc_id={doc_id}")
                eval_logger.info(f"text length chars={[len(t) for t in texts]}")
                eval_logger.info(f"image_inputs type={type(image_inputs)}")
                eval_logger.info(f"video_inputs type={type(video_inputs)}")

                if video_inputs is not None:
                    try:
                        eval_logger.info(
                            f"video_inputs len={len(video_inputs)}, "
                            f"first type={type(video_inputs[0])}, "
                            f"first shape={getattr(video_inputs[0], 'shape', None)}, "
                            f"first dtype={getattr(video_inputs[0], 'dtype', None)}"
                        )
                    except Exception as e:
                        eval_logger.warning(f"Could not inspect video_inputs: {e}")

                _debug_tensor_dict("PROCESSOR OUTPUT BEFORE DEVICE MOVE", inputs)
            # Device placement
            try:
                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)
            except Exception:
                eval_logger.exception("FAILED while moving processor inputs to device")
                _debug_tensor_dict("FAILED INPUTS STILL ON CPU", inputs)
                raise

            # Generation kwargs
            gen_kwargs = dict(all_gen_kwargs[0] or {})
            gen_kwargs.setdefault("max_new_tokens", 128)
            gen_kwargs["temperature"] = 0.0
            gen_kwargs["top_p"] = None
            gen_kwargs["num_beams"] = 1
            gen_kwargs["do_sample"] = False

            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            do_sample = False

            # Filter out keys not supported by model.generate()
            # LLaVA-OneVision-1.5 uses Qwen2_5_VLProcessor which produces second_per_grid_ts,
            # but the model's forward method does not accept this parameter
            unsupported_keys = ["second_per_grid_ts", "mm_token_type_ids"]
            filtered_inputs = {k: v for k, v in inputs.items() if k not in unsupported_keys}

            gen_args = {
                **filtered_inputs,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": pad_token_id,
                "num_beams": 1,
                "max_new_tokens": gen_kwargs["max_new_tokens"],
                "use_cache": self.use_cache,
                "do_sample": False,
            }

            _debug_tensor_dict("FILTERED INPUTS BEFORE GENERATE", filtered_inputs)
            eval_logger.info(f"gen_kwargs={gen_kwargs}")
            eval_logger.info(f"gen_args keys={list(gen_args.keys())}")
            eval_logger.info(f"model class={type(self.model)}")
            eval_logger.info(f"model device_map={getattr(self.model, 'hf_device_map', None)}")
            
            if do_sample:
                gen_args.update(
                    do_sample=True,
                    temperature=float(gen_kwargs.get("temperature", 1.0)),
                    top_p=float(gen_kwargs.get("top_p", 1.0)) if gen_kwargs.get("top_p") is not None else None,
                )

            generated_ids_trimmed = None
            try:
                start_time = time.time()
                with torch.inference_mode():
                    cont = self.model.generate(**gen_args)
                end_time = time.time()
                total_elapsed_time += end_time - start_time

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                total_tokens += sum(len(ids) for ids in generated_ids_trimmed)
            except Exception:
                eval_logger.exception("FAILED during model.generate")
                eval_logger.info(f"gen_args keys={list(gen_args.keys())}")
                raise

            text_outputs = self.tokenizer.batch_decode(
                generated_ids_trimmed if generated_ids_trimmed is not None else cont,
                skip_special_tokens=True,
            )

            for i, text_output in enumerate(text_outputs):
                raw_ans = text_output
                cleaned_ans = text_output.strip()

                if pred_mode:
                    self._init_pred_history()

                    current_doc_id = doc_id[i]
                    doc = self.task_dict[task][split][current_doc_id]

                    traj_id = str(doc.get("trajectory_id", doc.get("source_video_id", doc.get("id"))))
                    step_id = str(doc.get("step"))

                    user_msg = self._history_user_msg_from_doc(doc)
                    history_answer = self._format_answer_for_history(doc, cleaned_ans)

                    self._pred_history[traj_id][step_id] = (user_msg, history_answer)

                    self._dbg(
                        f"[PRED HISTORY STORE] traj_id={traj_id} step_id={step_id} "
                        f"answer='{history_answer[:160].replace(chr(10), ' ')}'"
                    )

                print("\n" + "=" * 80, flush=True)
                print("[RAW OUTPUT]", flush=True)
                print(raw_ans, flush=True)
                print("\n[CLEANED OUTPUT]", flush=True)
                print(cleaned_ans, flush=True)

                token_counts = TokenCounts(output_tokens=len(generated_ids_trimmed[i])) if generated_ids_trimmed is not None else None

                res.append(GenerationResult(text=cleaned_ans, token_counts=token_counts))
                self.cache_hook.add_partial("generate_until", (texts[i], gen_kwargs), cleaned_ans)
            pbar.update(1)


        if re_ords is not None:
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

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

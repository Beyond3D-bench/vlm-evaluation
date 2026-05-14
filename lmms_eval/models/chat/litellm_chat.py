"""LiteLLM chat backend with doc_to_messages + doc_to_visual video support."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Union

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.openai import OpenAICompatible as OpenAICompatibleChatBase
from lmms_eval.models.model_utils.usage_metrics import log_usage
from lmms_eval.models.simple.litellm import _PLACEHOLDER_API_KEY, _LiteLLMClientShim
from lmms_eval.protocol import ChatMessages


@register_model("litellm_chat")
class LiteLLMChatCompatible(OpenAICompatibleChatBase):
    """LiteLLM-backed chat backend that supports task doc_to_messages + doc_to_visual."""

    is_simple = False

    def __init__(
        self,
        model_version: str = "openai/gpt-4o-mini",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("OPENAI_API_BASE")

        # Force qwen_vl_utils to use decord for local video decoding.
        os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

        super().__init__(
            model_version=model_version,
            model=model,
            base_url=resolved_base_url,
            api_key=resolved_api_key or _PLACEHOLDER_API_KEY,
            azure_openai=False,
            **kwargs,
        )

        self.client = _LiteLLMClientShim(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        )

    def _debug_enabled(self) -> bool:
        return os.getenv("OOS_CHAT_DEBUG", "0") == "1"

    def _dbg(self, msg: str) -> None:
        if self._debug_enabled():
            print(msg, flush=True)

    @staticmethod
    def _ensure_content_list(content: Any) -> List[Dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            return list(content)
        raise TypeError(f"Unsupported message content type: {type(content)}")

    def _normalize_messages(self, raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(raw_messages, list) or len(raw_messages) == 0:
            raise ValueError("doc_to_messages returned empty or invalid messages")

        normalized: List[Dict[str, Any]] = []

        for msg in raw_messages:
            if not isinstance(msg, dict):
                raise TypeError(f"Message must be dict, got {type(msg)}")

            role = msg.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Invalid message role: {role}")

            content = self._ensure_content_list(msg.get("content"))

            # Avoid ChatMessages validation errors from text=None.
            cleaned_content = []
            for part in content:
                if part.get("type") == "text":
                    text = part.get("text")
                    if text is None:
                        text = ""
                    cleaned_content.append({"type": "text", "text": str(text)})
                else:
                    cleaned_content.append(part)

            normalized.append({"role": role, "content": cleaned_content})

        return normalized

    @staticmethod
    def _visuals_to_protocol_content(visuals: Any) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []

        if not visuals:
            return content

        for visual in visuals:
            if isinstance(visual, str) and visual.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv", ".webm")
            ):
                content.append({"type": "video", "url": visual})
            else:
                content.append({"type": "image", "url": visual})

        return content

    def _attach_visuals_to_first_user(
        self,
        messages: List[Dict[str, Any]],
        visual_content: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not visual_content:
            return messages

        for msg in messages:
            if msg.get("role") == "user":
                msg["content"] = visual_content + self._ensure_content_list(msg.get("content"))
                return messages

        messages.append({"role": "user", "content": visual_content})
        return messages

    def _build_video_kwargs(self) -> Dict[str, Any]:
        if self.video_fps is not None and self.video_fps > 0:
            return {
                "fps": self.video_fps,
                "video_reader_backend": "decord",
            }

        return {
            "nframes": self.max_frames_num,
            "video_reader_backend": "decord",
        }

    def _build_payload_for_request(self, req_args: tuple) -> dict:
        ctx, doc_to_source, gen_kwargs, doc_id, task, split = req_args
        doc = self.task_dict[task][split][doc_id]

        source_name = getattr(doc_to_source, "__name__", "")

        if source_name == "doc_to_messages" or source_name.endswith("doc_to_messages"):
            raw_messages = doc_to_source(doc)
            messages = self._normalize_messages(raw_messages)

            bound_task = getattr(doc_to_source, "__self__", None)
            if bound_task is None or not hasattr(bound_task, "doc_to_visual"):
                raise ValueError(
                    f"Could not access doc_to_visual from bound doc_to_messages for task={task}"
                )

            visuals = bound_task.doc_to_visual(doc)
            visual_content = self._visuals_to_protocol_content(visuals)
            messages = self._attach_visuals_to_first_user(messages, visual_content)

            self._dbg("\n" + "=" * 100)
            self._dbg(
                f"[LITELLM CHAT] doc_id={doc_id} "
                f"id={doc.get('id')} step={doc.get('step')}"
            )
            self._dbg(f"[LITELLM CHAT] visuals={visuals}")

            for i, msg in enumerate(messages):
                types = [
                    part.get("type")
                    for part in self._ensure_content_list(msg.get("content"))
                ]
                self._dbg(f"[MSG] idx={i} role={msg.get('role')} types={types}")

        else:
            # Fallback path for non-chat tasks.
            visuals = doc_to_source(doc)
            visual_content = self._visuals_to_protocol_content(visuals)
            messages = [
                {
                    "role": "user",
                    "content": visual_content + [{"type": "text", "text": str(ctx)}],
                }
            ]

        request_gen_kwargs = dict(gen_kwargs)
        max_new_tokens = min(request_gen_kwargs.get("max_new_tokens", 1024), 4096)
        temperature = request_gen_kwargs.get("temperature", 0)

        chat_messages = ChatMessages(messages=messages)

        payload = {
            "messages": chat_messages.to_openai_messages(
                video_kwargs=self._build_video_kwargs()
            ),
            "model": self.model_version,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

        # Reasoning/OpenAI-family special handling inherited from chat OpenAI pattern.
        if (
            "o1" in self.model_version
            or "o3" in self.model_version
            or "o4" in self.model_version
            or "gpt-5" in self.model_version
        ):
            payload.pop("temperature", None)
            payload.pop("max_tokens", None)
            payload["response_format"] = {"type": "text"}

        return payload

    def generate_until(self, requests) -> List[GenerationResult]:
        if not requests:
            return []

        reordered_requests = list(requests)
        responses: List[Union[GenerationResult, None]] = [None] * len(reordered_requests)

        pbar = tqdm(
            total=len(reordered_requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        for idx, req in enumerate(reordered_requests):
            started_at = time.time()

            try:
                payload = self._build_payload_for_request(req.args)
                response = self.client.chat.completions.create(**payload)

                elapsed = time.time() - started_at
                response_text = response.choices[0].message.content or ""

                input_tokens = 0
                output_tokens = len(response_text.split())
                reasoning_tokens = 0

                if hasattr(response, "usage") and response.usage:
                    input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(response.usage, "completion_tokens", 0) or output_tokens

                    details = getattr(response.usage, "completion_tokens_details", None)
                    if details:
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

                log_usage(
                    model_name=self.model_version,
                    task_name=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    source="model",
                )

                responses[idx] = GenerationResult(
                    text=response_text,
                    token_counts=TokenCounts(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                    ),
                )

            except Exception as exc:
                elapsed = time.time() - started_at
                error_msg = str(exc)
                eval_logger.error(f"LiteLLM chat request failed: {error_msg}")

                responses[idx] = GenerationResult(
                    text=f"[LMMS_EVAL_REQUEST_FAILED] {error_msg[:200]}",
                    token_counts=TokenCounts(),
                )

            pbar.update(1)

        pbar.close()
        return responses
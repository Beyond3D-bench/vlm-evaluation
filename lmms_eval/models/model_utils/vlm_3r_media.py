from typing import Any, Dict, List, Tuple


MediaItem = Tuple[str, str]
ChatTurn = Tuple[str, str]


def extract_ordered_media_and_turns(
    messages: List[Dict[str, Any]],
    visual_token: str,
) -> Tuple[List[MediaItem], List[ChatTurn], str]:
    """Preserve typed media order while converting message content to text turns."""
    media: List[MediaItem] = []
    system_lines: List[str] = []
    turns: List[ChatTurn] = []

    for message in messages:
        role = message.get("role", "user")
        text_parts: List[str] = []
        for part in message.get("content", []):
            if not isinstance(part, dict):
                continue
            content_type = part.get("type")
            if content_type == "text":
                text = str(part.get("text", "")).strip()
                if text:
                    text_parts.append(text)
                continue
            if content_type not in {"video", "image"}:
                continue
            url = part.get("url", part.get(content_type))
            if url:
                media.append((content_type, str(url)))
                text_parts.append(visual_token)

        text = "\n".join(text_parts).strip()
        if not text:
            continue
        if role == "system":
            system_lines.append(text)
        elif role in {"user", "assistant"}:
            turns.append((role, text))

    return media, turns, "\n\n".join(system_lines).strip()

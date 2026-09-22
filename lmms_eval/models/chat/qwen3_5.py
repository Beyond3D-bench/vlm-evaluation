from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.qwen3_vl_chat_fixed import Qwen3_VL_Chat_Fixed 
from lmms_eval.models.simple.qwen3_5 import Qwen3_5 as Qwen3_5_Simple


@register_model("qwen3_5_chat")
class Qwen3_5(Qwen3_VL_Chat_Fixed, Qwen3_5_Simple):
    """
    Qwen3.5 chat model -- uses Qwen3.5 defaults (via Qwen3_5_Simple)
    and chat generate_until (via Qwen3_VL_Chat).
    """

    def __init__(self, *args, enable_thinking=False, **kwargs):
        super().__init__(*args, enable_thinking=enable_thinking, **kwargs)

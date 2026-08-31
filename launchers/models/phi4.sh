#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-phi4_multimodal_chat_fixed}"
PHI4_PRETRAINED="${PHI4_CKPT:-microsoft/Phi-4-multimodal-instruct}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${PHI4_PRETRAINED},local_files_only=True,attn_implementation=sdpa}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-phi4_multimodal_chat_fixed}"

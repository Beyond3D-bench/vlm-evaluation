#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-internvl_hf_chat}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=OpenGVLab/InternVL3_5-38B-HF,,do_sample_frames=False,do_resize_video=False,low_cpu_mem_usage=True}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-internvl_hf_chat}"

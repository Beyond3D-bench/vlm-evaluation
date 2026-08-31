#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-internvl_hf}"
SENSENOVA_INTERNVL_PRETRAINED="${SENSENOVA_INTERNVL_CKPT:-sensenova/SenseNova-SI-1.5-InternVL3-8B}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${SENSENOVA_INTERNVL_PRETRAINED},trust_remote_code=True,local_files_only=True,fps=${OOS_SENSENOVA_INTERNVL_FPS:-${OOS_TARGET_FPS:-1}},num_frames=250,do_sample_frames=True,do_resize_video=False,low_cpu_mem_usage=True,attn_implementation=${SENSENOVA_INTERNVL_ATTN_IMPLEMENTATION:-eager}}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-sensenova_internvl}"

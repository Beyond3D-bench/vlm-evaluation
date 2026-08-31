#!/usr/bin/env bash
MODEL="${OOS_LMMS_MODEL:-internvl_hf}"
INTERNVL_PRETRAINED="${OOS_INTERNVL_PRETRAINED:-/cluster/scratch/fangma/hf_cache/models--OpenGVLab--InternVL3_5-8B-HF/snapshots/741a7d03020411e666c6109218ab71e08151ef86}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=${INTERNVL_PRETRAINED},do_sample_frames=True,num_frames=150,do_resize_video=False,low_cpu_mem_usage=True,local_files_only=True}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-internvl_hf}"

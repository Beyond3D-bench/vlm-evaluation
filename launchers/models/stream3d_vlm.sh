#!/usr/bin/env bash

export STREAM3D_VLM_REPO="${STREAM3D_VLM_REPO:-/work/courses/3dv/team1/Stream3D-VLM}"
if [[ -d "$STREAM3D_VLM_REPO/src" ]]; then
  export PYTHONPATH="$STREAM3D_VLM_REPO/src:${PYTHONPATH:-}"
fi

# TorchCodec dlopens Torch CUDA shared libraries directly; expose torch/lib from the active venv.
if command -v python >/dev/null 2>&1; then
  TORCH_LIB_DIR="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")' 2>/dev/null || true)"
  if [[ -n "$TORCH_LIB_DIR" && -d "$TORCH_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
  fi
fi

MODEL="${OOS_LMMS_MODEL:-stream3d_vlm}"
MODEL_ARGS="${OOS_MODEL_ARGS:-pretrained=JonnyYu828/Stream3D-VLM-4B,stream3d_repo=${STREAM3D_VLM_REPO},require_geometry=true,max_frames=250,video_decoder=torchcodec,attn_implementation=sdpa}"
OUTPUT_SUBDIR="${OOS_OUTPUT_SUBDIR:-stream3d_vlm}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"

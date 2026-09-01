# OOS model adapters

This fork retains only the chat backends selected by the model presets in
`launchers/models/`.

| Backend ID | Adapter | Launcher presets |
| --- | --- | --- |
| `cambrian_p` | `cambrian_p.py` | `cambrian_p` |
| `internvl_hf` | `internvl_hf.py` | `internvl` |
| `qwen3_5` | `qwen3_5.py` | `qwen3_6` |
| `qwen3_vl_chat_fixed` | `qwen3_vl_chat_fixed.py` | `qwen3_vl`, `sensenova_qwen` |
| `spatial_mllm` | `spatial_mllm.py` | `spatial_mllm` |
| `vlm_3r` | `vlm_3r.py` | `vlm3r` |

The small modules retained under `models/simple/` are implementation
dependencies of the fixed Qwen chat adapters. They are not separate public
backends in this evaluation repository.

External source repositories and compatibility patches are pinned in
`models/manifest.yaml`; see `models/README.md` for setup instructions.

# Compatibility patches

The evaluation uses public model source repositories at the revisions pinned in
`models/manifest.yaml`. These patches address environment and loading
compatibility; they do not change model architecture or learned weights.

Default evaluation patches:

- `vlm3r-low-memory-loading.patch`: loads the VLM-3R LoRA/fusion tensors into a
  meta-initialized model without excessive host-memory use.
- `vlm3r-cut3r-weight-path.patch`: permits node-local staging of the unchanged
  CUT3R checkpoint through `VLM3R_CUT3R_WEIGHTS`.
- `cut3r-curope-torch2-scalar-type.patch`: updates a deprecated PyTorch C++ API
  needed to build the CUDA RoPE extension.
- `cambrian-p-offline.patch`: avoids network-only auxiliary loads on offline
  compute nodes. It can be omitted when every upstream transitive dependency is
  available in the standard Hugging Face cache.

Before applying a patch, check it against the pinned clean source tree:

```bash
git -C /path/to/model apply --check /path/to/patch
git -C /path/to/model apply /path/to/patch
```

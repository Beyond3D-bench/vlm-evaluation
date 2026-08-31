# Reproducibility tools

These utilities prepare the external source repositories and Hugging Face
artifacts used by the OOS evaluation launchers.

- `bootstrap_model_sources.py` clones the official model repositories at the
  revisions in `models/manifest.yaml` and applies the declared compatibility
  patches.
- `download_hf_models.py` downloads pinned model artifacts into the standard
  Hugging Face cache on an Internet-connected machine.
- `verify_hf_offline.py` verifies that the pinned artifacts are available to an
  offline compute job.
- `model_manifest.py` provides the shared manifest parser used by these tools.

See [`../models/README.md`](../models/README.md) for the public setup workflow.

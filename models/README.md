# External model assets

`manifest.yaml` records the public source repositories, source revisions,
Hugging Face repositories, and checkpoint revisions used by the evaluation.
Model source trees and weights are intentionally not committed here.

Populate the persistent Hub cache on a machine with Internet access:

```bash
python tools/download_hf_models.py --model vlm3r
```

Verify the same assets from an offline compute environment:

```bash
python tools/verify_hf_offline.py --model vlm3r
```

Both commands refuse unpinned artifacts. A `null` revision in the manifest is
a visible cleanup item, not permission to silently use a moving `main` branch.


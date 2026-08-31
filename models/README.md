# External model assets

`manifest.yaml` records the public source repositories, source revisions,
Hugging Face repositories, and checkpoint revisions used by the evaluation.
Model source trees and weights are intentionally not committed here.

Prepare the official source repositories, their nested submodules, and the
declared compatibility patches:

```bash
python tools/bootstrap_model_sources.py --model vlm3r --model spatial_mllm
```

The default source destination is the gitignored `.model-sources/` directory.
Point launchers at it through `.env`:

```bash
OOS_MODEL_SOURCE_DIR="/absolute/path/to/oos_vlm_evaluation/.model-sources"
```

The command is idempotent: an already-applied patch is recognized rather than
applied twice. It refuses an unexpected remote, revision, or conflicting patch.
Verify an existing setup without changing it:

```bash
python tools/bootstrap_model_sources.py --verify-only --model vlm3r
```

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

<div align="center">

<h1>👀 Long Time No See:<br>Benchmarking VLMs for Out-of-Sight Spatiotemporal Reasoning in Egocentric Videos</h1>

<p>
Fangzhou Ma<sup>1*</sup> · Ivo Alexander Ban<sup>1*</sup> · Eren Homburg<sup>1*</sup> · Gabriele Goletto<sup>2</sup><br>
Rémi Pautrat<sup>2</sup> · Mahdi Rad<sup>2</sup> · Chiara Plizzari<sup>3</sup> · Marc Pollefeys<sup>1,2</sup>
</p>

<p><sup>1</sup> ETH Zurich · <sup>2</sup> Microsoft Spatial AI Lab · <sup>3</sup> Bocconi University<br>
<sup>*</sup> Equal contribution.</p>

<img src="docs/assets/teaser_video.gif" width="480" alt="An object being moved through a kitchen and leaving the camera view.">
<br>

</div>

<a id="benchmark"></a>

## 🧩 BEYOND3D

This repository provides the evaluation code for **BEYOND3D**, the benchmark introduced in our paper.
Built on **HD-EPIC** egocentric kitchen videos, it tests whether VLMs can track relocated objects
and reason about their last known spatial state after they leave view. The questions probe four capabilities: **visual grounding**, **temporal grounding**,
**scene localization**, and **3D spatial perception**. Models must recover when an object
was last visible or placed, identify its location in the scene, and estimate its direction
and distance relative to the camera or another object.

The evaluation set contains **8,000 out-of-sight questions** from 1,000 query anchors,
plus **1,000 visible controls** for the visibility question. Geometry-aware visibility
tracks and manual inspection establish that the out-of-sight targets are no longer
observable at query time.

<a id="results"></a>

## 📊 Results at a glance

Across the nine VLMs evaluated in the paper, the best model, **Qwen-3.6-27B**, reaches
**42.2% macro accuracy**, compared with **29.7% random guessing**. Recovering past events
and retaining updated spatial state remain challenging, especially as objects stay
out of sight for longer.

We report the mean accuracy across the eight question types, giving each type equal
weight. Models receive the video prefix up to the query time, with frame subsampling
where required by their context limits.

Explore question examples and detailed results on the
[project website](https://beyond3d-bench.github.io/website/).

<a id="quick-start"></a>

## 🛠️ Setup

Requires Linux x86-64, Python 3.10+, and an NVIDIA GPU with a CUDA 12.8-compatible driver.
The setup script uses `uv`; no Conda or root installation is needed. A C compiler
(`cc`) and `make` are required to build the bundled FFmpeg libraries.

```bash
git clone https://github.com/Zoulution/oos_vlm_evaluation.git
cd oos_vlm_evaluation

# Choose a directory with enough space for environments, checkpoints, and caches.
export OOS_STORAGE_ROOT="/absolute/path/to/storage"

# On a connected login/setup node: create the environment, download checkpoints,
# required model source, and the BEYOND3D dataset (vqa + preprocessed videos), then verify the installation.
bash setup.sh --model qwen3_5_9b
```

The command creates or reuses a locked environment for the selected model, downloads
its configured checkpoints and any required upstream source code, then checks the
installation and video decoding. It also downloads the BEYOND3D dataset snapshot
(currently about 2.6 GB) into `$HF_HUB_CACHE`, installs FFmpeg, and configures the
launchers to find it. Run setup on a machine with Internet access; evaluation never
downloads data.

Approximate checkpoint storage (environments, caches, datasets, and outputs are extra):

| Preset | Checkpoint | Model files |
| --- | --- | ---: |
| `qwen3_5_9b` | Qwen3.5-9B | 20 GB |
| `qwen3_6_27b` | Qwen3.6-27B | 56 GB |
| `qwen3_6_35b_a3b` | Qwen3.6-35B-A3B | 72 GB |
| `qwen3_vl` | Qwen3-VL-8B-Instruct | 18 GB |
| `internvl` | InternVL3.5-8B-HF | 18 GB |
| `sensenova_qwen` | SenseNova-SI-1.3-Qwen3-VL-8B | 18 GB |
| `cambrian_p` | Cambrian-P-7B and dependencies | 37 GB |
| `spatial_mllm` | Spatial-MLLM-v1.1 | 12 GB |
| `vlm3r` | VLM-3R and dependencies | 24 GB, including CUT3R |

At the current revisions, `--all` is expected to use about 279 GB: approximately
263 GB for unique Hugging Face model files, 3.6 GB for CUT3R and source repositories,
11.5 GB for the four hard-linked environments and `uv` cache, and 0.4 GB for the CUDA
toolchain. Reserve around 300 GB; datasets and evaluation outputs require additional
space. Actual usage can change when a manifest entry tracks an unpinned `main` revision.

## ▶️ Evaluate

```bash
# After setup, on a GPU compute node with the shared cache and checkpoints:
OOS_MODEL=qwen3_5_9b OOS_OFFLINE=1 bash launchers/run_oos_eval.sh
```

To use a local or derived dataset instead, set `OOS_DATASET_JSONL`; see the
[dataset format reference](docs/dataset.md).
To evaluate the temporal-cues variant shipped with BEYOND3D, set
`OOS_DATASET_FILE=vqa_temporal_cues.jsonl`.

Results: `outputs/oos_videoqa/<model>/` in the repository.
Questions run independently with `prefix` video context.

## 🤖 All models

```bash
bash setup.sh --all
```

Presets: `qwen3_5_9b`, `qwen3_6_27b`, `qwen3_6_35b_a3b`, `qwen3_vl`, `internvl`,
`sensenova_qwen`, `cambrian_p`, `spatial_mllm`, `vlm3r`.
Use the same preset for setup and `OOS_MODEL`; environments are selected automatically.

VLM-3R setup installs its pinned CUDA compiler in user storage. Its CUDA extension
is then built automatically on the first GPU run.

## 🖥️ Slurm

Adjust resources in [the job script](launchers/slurm_oos_eval.sh). After setup:

```bash
mkdir -p logs
OOS_MODEL=qwen3_5_9b OOS_LIMIT=2 OOS_OFFLINE=1 sbatch launchers/slurm_oos_eval.sh

# After setup.sh --all: test two questions per model.
OOS_OFFLINE=1 bash launchers/submit_oos_smoke_tests.sh --all --limit 2
```

Logs: `logs/oos_videoqa_<job-id>.{out,err}`. Omit `OOS_LIMIT` for a full single-model run.

## ⚙️ Configuration

Copy [.env.example](.env.example) to `.env`. Most users only need to set:

```bash
OOS_STORAGE_ROOT="/absolute/path/to/storage"
OOS_DATASET_JSONL="/absolute/path/to/questions.jsonl"
```

Other paths are derived automatically. Shared defaults live in
[launchers/oos_env.sh](launchers/oos_env.sh), while each file under
[launchers/models/](launchers/models/) defines how that model is launched.

## 📖 Citation

If you use this code or benchmark, please cite **Long Time No See**:

```bibtex
@misc{ma2026longtime,
  title  = {Long Time No See: Benchmarking VLMs for Out-of-Sight Spatiotemporal Reasoning in Egocentric Videos},
  author = {Ma, Fangzhou and Ban, Ivo Alexander and Homburg, Eren and Goletto, Gabriele and Pautrat, R\'emi and Rad, Mahdi and Plizzari, Chiara and Pollefeys, Marc},
  year   = {2026},
  note   = {Manuscript}
}
```

Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

## 🙏 Acknowledgements

BEYOND3D builds on the videos, object annotations, and scene reconstructions of
[HD-EPIC](https://hd-epic.github.io/site/).
Our evaluation code builds on [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).

## 📄 License

Our original contributions are licensed under [MIT](LICENSE).
Inherited lmms-eval code retains its MIT or Apache-2.0 license;
see [upstream notices](LICENSES/lmms-eval.txt).

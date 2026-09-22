---
license: cc-by-nc-4.0
task_categories:
  - visual-question-answering
tags:
  - video-question-answering
  - egocentric-video
  - spatial-reasoning
  - temporal-reasoning
  - benchmark
size_categories:
  - 1K<n<10K
---

# BEYOND3D

BEYOND3D is the evaluation dataset for **Long Time No See: Benchmarking VLMs
for Out-of-Sight Spatiotemporal Reasoning in Egocentric Videos**. It tests
whether vision-language models can retain an object's last known spatial state
after it leaves view in egocentric kitchen videos.

## Contents

```text
vqa_baseline.jsonl  # 2,000 trajectory records and 9,000 independent questions
vqa_temporal_cues.jsonl  # 1,000 trajectory records and 8,000 independent questions
videos/             # 156 preprocessed MP4 videos referenced by the JSONL
```

Each JSONL record contains either an eight-step out-of-sight trajectory or a
single visible-control question. `video_path` is a filename, relative to
`videos/`, so the dataset is portable across machines.

## Use with the evaluation code

On a connected setup machine, the [evaluation repository](https://github.com/Zoulution/oos_vlm_evaluation)
downloads this dataset, model checkpoints, and any required model source:

```bash
bash setup.sh --model qwen3_5_9b
```

The snapshot is cached in `$HF_HUB_CACHE`. Evaluation is cache-only and therefore
works on compute nodes without Internet access.

The default is `vqa_baseline.jsonl`. To use the temporal-cues variant, run the
same evaluation command with `OOS_DATASET_FILE=vqa_temporal_cues.jsonl`.

## License and attribution

The videos, object annotations, and scene reconstructions are derived from
[HD-EPIC](https://hd-epic.github.io/site/), which is released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This dataset
is therefore for non-commercial use only. Please retain this license notice,
give appropriate credit to HD-EPIC, link to the license, and indicate any
changes.

Please cite both HD-EPIC and Long Time No See when using this dataset.

```bibtex
@InProceedings{perrett2025hdepic,
  author    = {Perrett, Toby and Darkhalil, Ahmad and Sinha, Saptarshi and Emara, Omar and Pollard, Sam and Parida, Kranti and Liu, Kaiting and Gatti, Prajwal and Bansal, Siddhant and Flanagan, Kevin and Chalk, Jacob and Zhu, Zhifan and Guerrier, Rhodri and Abdelazim, Fahd and Zhu, Bin and Moltisanti, Davide and Wray, Michael and Doughty, Hazel and Damen, Dima},
  title     = {HD-EPIC: A Highly-Detailed Egocentric Video Dataset},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}

@misc{ma2026longtime,
  title  = {Long Time No See: Benchmarking VLMs for Out-of-Sight Spatiotemporal Reasoning in Egocentric Videos},
  author = {Ma, Fangzhou and Ban, Ivo Alexander and Homburg, Eren and Goletto, Gabriele and Pautrat, R\'emi and Rad, Mahdi and Plizzari, Chiara and Pollefeys, Marc},
  year   = {2026},
  note   = {Manuscript}
}
```

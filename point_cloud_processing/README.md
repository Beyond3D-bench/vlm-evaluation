# Kitchen / Egocentric Point Cloud Processing Scripts

These scripts are meant for raw reconstructed `.ply` point clouds from CUT3R / VGGT / similar dense pointmap pipelines.

## Install

```bash
python -m venv .venv-pcd
source .venv-pcd/bin/activate
pip install -r requirements.txt
```

## VLM-3R Debug Export Patches

After cloning a fresh `VLM-3R` repo next to this `lmms-eval` checkout, apply the
local compatibility/debug patches from the `lmms-eval` repo:

```bash
cd /work/courses/3dv/team1/VLM-3R
git apply /work/courses/3dv/team1/lmms-eval/patches/vlm-3r-point-cloud-export-path.patch

cd /work/courses/3dv/team1/VLM-3R/CUT3R
git apply /work/courses/3dv/team1/lmms-eval/patches/cut3r-curope-torch2-scalar-type.patch
```

Then rebuild the CUT3R RoPE extension in the environment used for VLM-3R:

```bash
cd /work/courses/3dv/team1/VLM-3R/CUT3R/src/croco/models/curope
python setup.py build_ext --inplace
```

The first patch lets `lmms-eval` pass `.ply` output paths into VLM-3R/CUT3R.
The second patch fixes CUT3R's CUDA extension build with newer PyTorch by using
`tokens.scalar_type()` instead of the deprecated `tokens.type()`.


## 1. Process one already-merged PLY

Start with aggressive downsampling and outlier removal:

```bash
python process_pointcloud.py \
  --input raw_kitchen.ply \
  --out_dir processed \
  --voxel 0.03 \
  --sor_neighbors 30 \
  --sor_std 2.0 \
  --make_cuts \
  --make_slices
```

This creates:

```text
processed/
  00_raw_copy.ply
  01_downsampled.ply
  02_clean_sor.ply
  03_crop_quantile.ply
  cuts/
  slices/
```

If the cloud still looks closed, inspect the files in `cuts/` and `slices/`.

## 2. Crop harder

Example: remove the outer 10% along X/Y/Z:

```bash
python process_pointcloud.py \
  --input raw_kitchen.ply \
  --out_dir processed_crop \
  --voxel 0.03 \
  --crop_x 0.10 0.90 \
  --crop_y 0.05 0.95 \
  --crop_z 0.02 0.98 \
  --make_cuts \
  --make_slices
```

## 3. Visualize

```bash
python visualize_ply.py processed/02_clean_sor.ply --point_size 1.0
```

You can also visualize multiple files:

```bash
python visualize_ply.py processed/cuts/cut_x_keep_low_75.ply processed/cuts/cut_y_keep_mid_20_80.ply
```

## 3b. Boost dark PLY colors

If the exported PLY has RGB fields but looks nearly black/gray, stretch the
stored vertex colors before visualizing:

```bash
python boost_ply_colors.py \
  --input processed/final_processed.ply \
  --output processed/final_processed_color_boosted.ply \
  --low 1 \
  --high 99.5 \
  --gamma 0.65 \
  --brightness 1.25

python visualize_ply.py processed/final_processed_color_boosted.ply --point_size 1.0
```

## 4. Fuse per-frame point clouds more cleanly

If you have one PLY per frame, use temporal stability. This removes points that only appear briefly, such as hands/tools.

```bash
python stable_fuse_frames.py \
  --input_dir frame_pointclouds \
  --out stable_fused.ply \
  --voxel 0.03 \
  --min_frames 3 \
  --max_frames 200
```

For a static kitchen, increase `--min_frames` to keep only voxels observed in several frames:

```bash
python stable_fuse_frames.py \
  --input_dir frame_pointclouds \
  --out stable_background.ply \
  --voxel 0.03 \
  --min_frames 5
```

## Suggested order for your current case

```bash
python process_pointcloud.py --input YOUR_FILE.ply --out_dir pcd_processed --voxel 0.05 --make_cuts --make_slices
python process_pointcloud.py --input YOUR_FILE.ply --out_dir pcd_processed_fine --voxel 0.02 --sor_neighbors 30 --sor_std 2.0 --make_cuts --make_slices
python visualize_ply.py pcd_processed/cuts/cut_x_keep_low_75.ply --point_size 1.0
```

If all views still look like a closed blob, the issue is probably in fusion: either dynamic foreground is being merged, or frame-local pointmaps were accumulated without transforming into world coordinates.

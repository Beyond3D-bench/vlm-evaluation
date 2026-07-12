# OOS VLM cu128 Apptainer Image

Build this image on a machine where Docker/Podman and Apptainer are allowed.
ETH Euler docs say custom container creation is not permitted directly on Euler;
build locally, then copy the `.sif` file to `$SCRATCH`.

## Build

From the repository root:

```bash
docker build -f containers/Dockerfile.oos-cu128 -t oos-vlm-cu128 .
apptainer build oos-vlm-cu128.sif docker-daemon://oos-vlm-cu128:latest
```

If your local machine has Docker but not Apptainer:

```bash
docker build -f containers/Dockerfile.oos-cu128 -t oos-vlm-cu128 .
docker save oos-vlm-cu128:latest -o oos-vlm-cu128.tar
```

Then move the tarball to a machine with Apptainer and run:

```bash
apptainer build oos-vlm-cu128.sif docker-archive://oos-vlm-cu128.tar
```

## Copy To Euler

```bash
ssh fangma@euler.ethz.ch 'mkdir -p $SCRATCH/containers'
scp oos-vlm-cu128.sif fangma@euler.ethz.ch:/cluster/scratch/fangma/containers/
```

## Test On Euler

```bash
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer"
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$APPTAINER_CACHEDIR"

apptainer exec --nv \
  --bind /cluster/home/fangma:/cluster/home/fangma \
  --bind /cluster/scratch/fangma:/cluster/scratch/fangma \
  "$SCRATCH/containers/oos-vlm-cu128.sif" \
  python3.10 -c 'import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__, torch.cuda.is_available())'
```

## Run The Repo Command

Keep the repo and datasets on Euler, but run Python from the container:

```bash
cd /cluster/home/fangma/oos_vlm_evaluation
apptainer exec --nv \
  --bind /cluster/home/fangma:/cluster/home/fangma \
  --bind /cluster/scratch/fangma:/cluster/scratch/fangma \
  "$SCRATCH/containers/oos-vlm-cu128.sif" \
  python3.10 launchers/run_lmms_eval_with_trace.py --help
```


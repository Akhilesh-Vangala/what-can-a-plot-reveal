#!/bin/bash
# One-time setup on HPC Cloud Bursting (run from an OOD terminal, ideally on a CPU node).
# Creates a Singularity overlay with Miniforge + the project's Python packages.
set -euo pipefail

BASE=${SCRATCH:-$HOME}/visml
IMG=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif
OVL=$BASE/overlay-25GB-500K.ext3

mkdir -p "$BASE"/{runs,hf_cache,logs}
if [ ! -f "$OVL" ]; then
  cp /share/apps/overlay-fs-ext3/overlay-25GB-500K.ext3.gz "$BASE/"
  gunzip "$OVL.gz"
fi

singularity exec --overlay "$OVL":rw "$IMG" /bin/bash -c '
set -e
if [ ! -d /ext3/miniforge3 ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
  bash /tmp/mf.sh -b -p /ext3/miniforge3
fi
cat > /ext3/env.sh <<EOS
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:\$PATH
EOS
source /ext3/env.sh
pip install -q --upgrade pip
pip install -q torch torchvision numpy pandas matplotlib scikit-learn statsmodels pyreadr rdata
python -c "import torch, sklearn; print(\"installed torch\", torch.__version__, \"sklearn\", sklearn.__version__)"
'
echo "Done. Overlay: $OVL"

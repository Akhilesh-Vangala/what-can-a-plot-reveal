#!/bin/bash
# One-time: add a Python 3.12 + vLLM environment inside the existing overlay (run in an OOD Jupyter terminal, CPU node).
set -euo pipefail
BASE=${SCRATCH:-$HOME}/visml
IMG=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif
singularity exec --overlay $BASE/overlay-25GB-500K.ext3:rw $IMG /bin/bash -c '
set -e
source /ext3/env.sh
[ -d /ext3/envs/vllm ] || conda create -y -q -p /ext3/envs/vllm python=3.12
/ext3/envs/vllm/bin/pip install -q vllm scikit-learn
/ext3/envs/vllm/bin/python -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__)"
'
echo "Done."

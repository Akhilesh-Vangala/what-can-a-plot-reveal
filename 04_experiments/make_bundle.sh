#!/bin/bash
# Build the single upload for the cluster: visml_bundle.zip (all code + all job scripts + VERSION stamp).
#   bash make_bundle.sh   -> 04_experiments/visml_bundle.zip
set -euo pipefail
cd "$(dirname "$0")"
PY=/Applications/Anaconda/anaconda3/bin/python
$PY hpc/make_jobs.py
for f in hpc/*.sbatch hpc/run_stage.sh; do bash -n "$f"; done
$PY -m py_compile code/run_v4.py code/interventions.py code/make_positive.py code/rw_transfer.py code/score_confirm.py hpc/vlm_read.py hpc/b2_v4.py code/plotreveal/*.py
STAGE=$(mktemp -d)
cp -R code/plotreveal code/run_v4.py code/interventions.py code/make_positive.py code/rw_transfer.py code/score_confirm.py "$STAGE/"
mkdir "$STAGE/hpc" && cp hpc/*.sbatch hpc/run_stage.sh hpc/vlm_read.py hpc/b2_v4.py "$STAGE/hpc/"
find "$STAGE" -name "__pycache__" -prune -exec rm -rf {} +
HASH=$(cd "$STAGE" && find . -type f | sort | xargs cat | shasum | cut -c1-8)
echo "$(date +%Y-%m-%d_%H%M)-$HASH" > "$STAGE/VERSION"
rm -f visml_bundle.zip && (cd "$STAGE" && zip -rq "$OLDPWD/visml_bundle.zip" .)
rm -rf "$STAGE"
echo "built visml_bundle.zip — VERSION $(unzip -p visml_bundle.zip VERSION)"
unzip -l visml_bundle.zip | tail -n +4 | awk '{print $4}' | grep -v "^$" | head -40

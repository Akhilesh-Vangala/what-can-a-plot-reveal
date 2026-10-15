#!/bin/bash
# The ONE launcher. Upload visml_bundle.zip to ~/visml_hpc, then in an OOD terminal:
#   unzip -oq ~/visml_hpc/visml_bundle.zip -d $SCRATCH/visml/code && bash $SCRATCH/visml/code/hpc/run_stage.sh <stage>
# Stages:
#   check                       inputs + queue
#   interventions [RUN]         badges/companions + 4 readers on an existing run (default v4_synthetic)
#   confirm                     the preregistered confirmatory study (fresh seeds, synthetic + real, all readers,
#                               controls, interventions) — everything chained, one command
#   positive                    positive control (blatant cases) + value probe on confirm_synthetic, 5 readers
#   realworld                   real-world check: unzip ~/visml_hpc/realworld_data.zip, transfer job + 5 readers
#   rtr                         read-then-reason: extraction items on 3 sets + 5 readers; proportion-based transfer
#   gaps                        annotated machine-reader designs + 5 readers; multi-plot benchmark (both runs)
# Every submission: code version printed; inputs checked first; jobs requeue if a node dies; a failed job cancels the
# jobs that depend on it (--kill-on-invalid-dep) instead of leaving them waiting forever.
set -euo pipefail
BASE=${SCRATCH:-$HOME}/visml
CODE=$BASE/code
H=$CODE/hpc
cd ~/visml_hpc
echo "code version: $(cat $CODE/VERSION)"
SB="sbatch --parsable --kill-on-invalid-dep=yes"
Q7=Qwen/Qwen2.5-VL-7B-Instruct; I8=OpenGVLab/InternVL3-8B; LV=llava-hf/llava-onevision-qwen2-7b-ov-hf
Q32=Qwen/Qwen2.5-VL-32B-Instruct-AWQ; I38=OpenGVLab/InternVL3-38B-AWQ
MISS=0
ok() { [ -e "$1" ] && echo "  ok      $1" || { echo "  MISSING $1"; MISS=1; }; }
check_env() {
  ok $BASE/overlay-25GB-500K.ext3; ok /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif
  ok $BASE/hf_cache; ok $CODE/run_v4.py; ok $CODE/interventions.py; ok $H/vlm_read.py; ok $H/b2_v4.py
  for j in cpu b2 readers iv_cpu; do ok $H/$j.sbatch; done
}
check_run() {
  local R=$CODE/runs/$1
  for f in meta.json results.json b2_results.json manifest.jsonl img reader_items_variants.json reader_items_core.json; do ok $R/$f; done
}

case "${1:-check}" in
  check)
    echo "environment:"; check_env
    for r in v4_synthetic v4_magic; do echo "run $r:"; check_run $r; done
    squeue -u $USER ;;
  interventions)
    RUN=${2:-v4_synthetic}
    echo "preflight:"; check_env; check_run $RUN
    [ $MISS = 0 ] || { echo "Not submitting: inputs missing (see above)."; exit 1; }
    C=$($SB $H/iv_cpu.sbatch $RUN)
    G=$($SB --dependency=afterok:$C $H/readers.sbatch $RUN reader_items_interventions.json:iv:$Q7 \
        reader_items_interventions.json:iv:$I8 reader_items_interventions.json:iv:$Q32 reader_items_interventions.json:iv:$LV)
    echo "submitted interventions on $RUN: CPU $C -> readers $G" ;;
  confirm)
    echo "preflight:"; check_env
    [ $MISS = 0 ] || { echo "Not submitting: environment incomplete (see above)."; exit 1; }
    [ ! -e $CODE/runs/confirm_synthetic/results.json ] || { echo "confirm_synthetic already exists — not overwriting."; exit 1; }
    S=confirm_synthetic; M=confirm_magic
    C1=$($SB $H/cpu.sbatch $S synthetic confirm confirm)
    C2=$($SB $H/cpu.sbatch $M magic confirm confirm)
    B1=$($SB --dependency=afterok:$C1 $H/b2.sbatch $S)
    B2=$($SB --dependency=afterok:$C2 $H/b2.sbatch $M)
    R1=$($SB --dependency=afterok:$C1 $H/readers.sbatch $S reader_items.json:-:$Q7 reader_items.json:-:$I8 \
         reader_items.json:-:$LV reader_items_controls.json:controls:$Q7 reader_items_controls.json:controls:$I8)
    R2=$($SB --dependency=afterok:$C1 $H/readers.sbatch $S reader_items.json:-:$Q32 \
         reader_items_controls.json:controls:$Q32 reader_items_core.json:core:$I38)
    R3=$($SB --dependency=afterok:$C2 $H/readers.sbatch $M reader_items_core.json:core:$Q7 \
         reader_items_core.json:core:$I8 reader_items_core.json:core:$Q32)
    V1=$($SB --dependency=afterok:$C1:$B1 $H/iv_cpu.sbatch $S)
    V2=$($SB --dependency=afterok:$V1 $H/readers.sbatch $S reader_items_interventions.json:iv:$Q7 \
         reader_items_interventions.json:iv:$I8 reader_items_interventions.json:iv:$Q32 reader_items_interventions.json:iv:$LV)
    echo "submitted confirmatory study:"
    echo "  synthetic: CPU $C1 -> B2 $B1, readers $R1 (fast+controls) $R2 (32B+controls+38B), interventions $V1 -> $V2"
    echo "  real:      CPU $C2 -> B2 $B2, readers $R3" ;;
  positive)
    echo "preflight:"; check_env; ok $CODE/make_positive.py; ok $H/pos_cpu.sbatch
    for f in manifest.jsonl meta.json reader_items_core.json img; do ok $CODE/runs/confirm_synthetic/$f; done
    [ $MISS = 0 ] || { echo "Not submitting: inputs missing (see above)."; exit 1; }
    [ ! -e $CODE/runs/positive/truth.json ] || { echo "runs/positive already exists — not overwriting."; exit 1; }
    P=$($SB $H/pos_cpu.sbatch)
    G1=$($SB --dependency=afterok:$P $H/readers.sbatch positive reader_items.json:-:$Q7 reader_items.json:-:$I8 \
         reader_items.json:-:$LV reader_items.json:-:$Q32 reader_items.json:-:$I38)
    G2=$($SB --dependency=afterok:$P $H/readers.sbatch confirm_synthetic reader_items_probe.json:probe:$Q7 \
         reader_items_probe.json:probe:$I8 reader_items_probe.json:probe:$LV reader_items_probe.json:probe:$Q32 \
         reader_items_probe.json:probe:$I38)
    echo "submitted positive control: CPU $P -> readers $G1 (positive, 5 models), $G2 (value probe, 5 models)" ;;
  realworld)
    echo "preflight:"; check_env; ok $CODE/rw_transfer.py; ok $H/rw_cpu.sbatch; ok ~/visml_hpc/realworld_data.zip
    for f in features.npz results.json meta.json; do ok $CODE/runs/confirm_synthetic/$f; done
    [ $MISS = 0 ] || { echo "Not submitting: inputs missing (see above)."; exit 1; }
    [ ! -e $CODE/runs/realworld/transfer.json ] || { echo "runs/realworld already scored — not overwriting."; exit 1; }
    unzip -oq ~/visml_hpc/realworld_data.zip -d $CODE/runs/
    for f in truth.json cases.json features.npz reader_items.json img; do ok $CODE/runs/realworld/$f; done
    [ $MISS = 0 ] || { echo "Not submitting: realworld data incomplete after unzip."; exit 1; }
    echo "  images: $(ls $CODE/runs/realworld/img | wc -l) (expect 3960)"
    T=$($SB $H/rw_cpu.sbatch)
    G1=$($SB $H/readers.sbatch realworld reader_items.json:-:$Q7 reader_items.json:-:$I8 reader_items.json:-:$LV)
    G2=$($SB $H/readers.sbatch realworld reader_items.json:-:$Q32 reader_items.json:-:$I38)
    echo "submitted real-world check: transfer $T, readers $G1 (7B/8B/LLaVA), $G2 (32B/38B)" ;;
  rtr)
    echo "preflight:"; check_env; ok $CODE/rtr.py; ok $CODE/rtr_prepare.py; ok $H/rtr_cpu.sbatch
    for f in confirm_synthetic/features.npz confirm_synthetic/img positive/img positive/reader_items.json realworld/img realworld/features.npz; do ok $CODE/runs/$f; done
    [ $MISS = 0 ] || { echo "Not submitting: inputs missing (see above)."; exit 1; }
    [ ! -e $CODE/runs/confirm_synthetic/rtr_values.json ] || { echo "read-then-reason already prepared — not overwriting."; exit 1; }
    P=$($SB $H/rtr_cpu.sbatch)
    J=""
    for RUN in confirm_synthetic positive realworld; do
      A=$($SB --dependency=afterok:$P $H/readers.sbatch $RUN reader_items_extract.json:extract:$Q7 reader_items_extract.json:extract:$I8 reader_items_extract.json:extract:$LV)
      B=$($SB --dependency=afterok:$P $H/readers.sbatch $RUN reader_items_extract.json:extract:$Q32 reader_items_extract.json:extract:$I38)
      J="$J $RUN:$A,$B"
    done
    echo "submitted read-then-reason: prep $P -> readers$J" ;;
  gaps)
    echo "preflight:"; check_env; ok $CODE/annotate.py; ok $CODE/gaps_prepare.py; ok $H/ann_cpu.sbatch; ok $H/sets_cpu.sbatch
    for f in confirm_synthetic/features.npz confirm_synthetic/reader_items_core.json confirm_magic/features.npz; do ok $CODE/runs/$f; done
    [ $MISS = 0 ] || { echo "Not submitting: inputs missing (see above)."; exit 1; }
    [ ! -e $CODE/runs/confirm_synthetic/reader_items_ann.json ] || { echo "annotated items already exist — not overwriting."; exit 1; }
    A=$($SB $H/ann_cpu.sbatch)
    G1=$($SB --dependency=afterok:$A $H/readers.sbatch confirm_synthetic reader_items_ann.json:ann:$Q7 reader_items_ann.json:ann:$I8 reader_items_ann.json:ann:$LV)
    G2=$($SB --dependency=afterok:$A $H/readers.sbatch confirm_synthetic reader_items_ann.json:ann:$Q32 reader_items_ann.json:ann:$I38)
    S=$($SB $H/sets_cpu.sbatch)
    echo "submitted gaps: annotated designs $A -> readers $G1, $G2; multi-plot benchmark $S" ;;
  *) echo "unknown stage $1"; exit 1 ;;
esac
squeue -u $USER

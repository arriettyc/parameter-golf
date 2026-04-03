#!/bin/bash
# Usage: sh run_train.sh [run_id] [perf_on] [nsys_steps] [train_script]
#   run_id       : name for this run (default: baseline_sp1024)
#   perf_on      : 1/y/yes to enable CUDA event timing + MFU/MBU (disables compile), 0 otherwise (default: 0)
#   nsys_steps   : N>0 to capture N steps with Nsight Systems via cudaProfilerApi (default: 0)
#   train_script : path to training script (default: train_gpt.py)
#
# Examples:
#   sh run_train.sh                          # normal training
#   sh run_train.sh my_run                   # normal training, custom run_id
#   sh run_train.sh my_run 1                 # PERF_METRICS mode (eager, slow)
#   sh run_train.sh my_run 0 5              # nsys capture of 5 steps (compiled, fast)
#     → then open logs/my_run.nsys-rep in Nsight Systems
#   sh run_train.sh xin_exp 0 0 records/track_10min_16mb/2026-04-02_xin/train_gpt_xin.py

RUN_ID=${1:-baseline_sp1024}

# Normalize perf flag: treat 1/y/Y/yes/true as 1, everything else as 0
_PERF_RAW=${2:-0}
case "$_PERF_RAW" in
  1|y|Y|yes|YES|true|TRUE) PERF_METRICS=1 ;;
  *) PERF_METRICS=0 ;;
esac

NSYS_STEPS=${3:-0}
TRAIN_SCRIPT=${4:-train_gpt.py}

TORCHRUN=$(which torchrun 2>/dev/null || echo "/home/xin/parameter-golf/.venv/bin/torchrun")

TRAIN_CMD="PERF_METRICS=$PERF_METRICS \
NSYS_STEPS=$NSYS_STEPS \
RUN_ID=$RUN_ID \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
$TORCHRUN --standalone --nproc_per_node=1 $TRAIN_SCRIPT"

if [ "$NSYS_STEPS" -gt 0 ] 2>/dev/null; then
    # Stop training right after the capture window ends (3 warmup + NSYS_STEPS)
    STOP_ITER=$((3 + NSYS_STEPS))
    echo "nsys mode: capturing $NSYS_STEPS steps → logs/${RUN_ID}.nsys-rep (training stops at step $STOP_ITER)"
    sudo -E nsys profile \
        --trace=cuda,nvtx,osrt \
        --trace-fork-before-exec=true \
        --cuda-flush-interval=0 \
        --gpu-metrics-devices=all \
        --output="logs/${RUN_ID}" \
        --force-overwrite=true \
        sh -c "ITERATIONS=$STOP_ITER VAL_LOSS_EVERY=0 $TRAIN_CMD"
else
    eval "$TRAIN_CMD"
fi

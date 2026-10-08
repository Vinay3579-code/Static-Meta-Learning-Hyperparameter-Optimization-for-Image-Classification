#!/usr/bin/env bash

set -euo pipefail

cd /home/drive3/ml_dl_cp
source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ANCHORS="configs/search_spaces/oracle_anchors_40.json"
TRAIN_EPISODES=100
VALIDATION_EPISODES=100
MODEL_SEEDS=(101 202 303)

mkdir -p results/oracles/robustness
mkdir -p results/logs

run_task() {
    local dataset="$1"
    local task_id="$2"
    local manifest="$3"
    local data_root="$4"

    local output="results/oracles/robustness/${task_id}.csv"
    local best="results/oracles/robustness/${task_id}_best.json"
    local log="results/logs/${task_id}_robustness.log"

    echo
    echo "============================================================"
    echo "Running ${task_id}"
    echo "============================================================"

    python -u scripts/run_oracle_sweep.py \
      --task-manifest "${manifest}" \
      --task-id "${task_id}" \
      --anchors "${ANCHORS}" \
      --data-root "${data_root}" \
      --output "${output}" \
      --best-output "${best}" \
      --train-episodes "${TRAIN_EPISODES}" \
      --validation-episodes "${VALIDATION_EPISODES}" \
      --model-seeds "${MODEL_SEEDS[@]}" \
      --overwrite \
      2>&1 | tee "${log}"

    python scripts/analyze_oracle_sweep.py \
      --input "${output}" \
      --ranked-output \
        "results/oracles/robustness/${task_id}_ranked.csv" \
      --summary-output \
        "results/oracles/robustness/${task_id}_summary.json" \
      --equivalence-epsilon 0.002 \
      --softmax-temperature 0.01
}

run_task \
  "omniglot" \
  "omniglot_train_0000" \
  "data/manifests/tasks/omniglot_train_dev.json" \
  "data/raw/omniglot"

run_task \
  "cifar100" \
  "cifar100_train_0000" \
  "data/manifests/tasks/cifar100_train_dev.json" \
  "data/raw/cifar100"

run_task \
  "miniimagenet" \
  "miniimagenet_train_0000" \
  "data/manifests/tasks/miniimagenet_train_dev.json" \
  "data/raw/miniimagenet"

echo
echo "ROBUST THREE-DATASET ORACLE SWEEPS: PASS"

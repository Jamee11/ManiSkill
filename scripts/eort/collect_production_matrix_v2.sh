#!/usr/bin/env bash
set -euo pipefail

# Orchestrate the production Push/Pick x train/val/test matrix through the
# already-validated single-shard collector. Print-only unless EXECUTE=1.
ROOT=${MANISKILL_EORT_COLLECTION_ROOT:-/remote-home/jinminghao/datasets/maniskill_eort_large_v2}
TASKS=${MANISKILL_EORT_MATRIX_TASKS:-push_cube,pick_cube}
SPLITS=${MANISKILL_EORT_MATRIX_SPLITS:-train,val,test}
EXECUTE=${MANISKILL_EORT_MATRIX_EXECUTE:-0}
INCLUDE_GEOMETRY=${MANISKILL_EORT_MATRIX_INCLUDE_GEOMETRY_TEST:-1}
TRAIN_TRAJ=${MANISKILL_EORT_MATRIX_TRAIN_TRAJ:-500}
VAL_TRAJ=${MANISKILL_EORT_MATRIX_VAL_TRAJ:-100}
TEST_TRAJ=${MANISKILL_EORT_MATRIX_TEST_TRAJ:-200}
GEOMETRY_TRAJ=${MANISKILL_EORT_MATRIX_GEOMETRY_TEST_TRAJ:-200}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

[[ "${EXECUTE}" == 0 || "${EXECUTE}" == 1 ]] || { echo "MANISKILL_EORT_MATRIX_EXECUTE must be 0 or 1" >&2; exit 2; }
[[ "${INCLUDE_GEOMETRY}" == 0 || "${INCLUDE_GEOMETRY}" == 1 ]] || { echo "MANISKILL_EORT_MATRIX_INCLUDE_GEOMETRY_TEST must be 0 or 1" >&2; exit 2; }
if [[ "${EXECUTE}" == 1 && -z "${MANISKILL_EORT_CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set MANISKILL_EORT_CUDA_VISIBLE_DEVICES before execution" >&2
  exit 2
fi

run_shard() {
  MANISKILL_EORT_COLLECTION_ROOT="${ROOT}" \
  MANISKILL_EORT_TASK="$1" \
  MANISKILL_EORT_SPLIT="$2" \
  MANISKILL_EORT_VARIANTS="$3" \
  MANISKILL_EORT_START_SEED="$4" \
  MANISKILL_EORT_ACTION_SOURCE=metric_task_delta_pose \
  MANISKILL_EORT_INCLUDE_FUTURE_TARGETS=1 \
  NUM_TRAJ="$5" \
  DRY_RUN=$((1 - EXECUTE)) \
  bash "${SCRIPT_DIR}/collect_large_objectcentric_v2.sh"
}

IFS=',' read -r -a tasks <<< "${TASKS}"
IFS=',' read -r -a splits <<< "${SPLITS}"
for task in "${tasks[@]}"; do
  [[ "${task}" == push_cube || "${task}" == pick_cube ]] || { echo "Unknown matrix task: ${task}" >&2; exit 2; }
  for split in "${splits[@]}"; do
    case "${split}" in
      train) run_shard "${task}" train fixed,camera_rand,occluded 0 "${TRAIN_TRAJ}" ;;
      val) run_shard "${task}" val fixed,camera_rand,occluded 1000000 "${VAL_TRAJ}" ;;
      test)
        run_shard "${task}" test fixed,camera_rand,occluded 2000000 "${TEST_TRAJ}"
        [[ "${INCLUDE_GEOMETRY}" == 0 ]] || run_shard "${task}" test geometry_rand 3000000 "${GEOMETRY_TRAJ}"
        ;;
      *) echo "Unknown matrix split: ${split}" >&2; exit 2 ;;
    esac
  done
done

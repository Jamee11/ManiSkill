#!/usr/bin/env bash
set -euo pipefail

# Print the six-task/split production schedule by default; set EXECUTE=1 to run.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${MANISKILL_EORT_COLLECTION_ROOT:-/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real}
TASKS=${MANISKILL_EORT_MATRIX_TASKS:-push_cube,pick_cube}
SPLITS=${MANISKILL_EORT_MATRIX_SPLITS:-train,val,test}
EXECUTE=${MANISKILL_EORT_MATRIX_EXECUTE:-0}
[[ "${EXECUTE}" == 0 || "${EXECUTE}" == 1 ]] || { echo "MANISKILL_EORT_MATRIX_EXECUTE must be 0 or 1" >&2; exit 2; }

IFS=',' read -r -a tasks <<< "${TASKS}"
IFS=',' read -r -a splits <<< "${SPLITS}"
for task in "${tasks[@]}"; do
  for split in "${splits[@]}"; do
    case "${split}" in
      train) count=${MANISKILL_EORT_MATRIX_TRAIN_TRAJ:-500}; seed=0 ;;
      val) count=${MANISKILL_EORT_MATRIX_VAL_TRAJ:-100}; seed=1000000 ;;
      test) count=${MANISKILL_EORT_MATRIX_TEST_TRAJ:-200}; seed=2000000 ;;
      *) echo "Unknown split ${split}" >&2; exit 2 ;;
    esac
    MANISKILL_EORT_COLLECTION_ROOT="${ROOT}" MANISKILL_EORT_TASK="${task}" \
      MANISKILL_EORT_SPLIT="${split}" MANISKILL_EORT_START_SEED="${seed}" NUM_TRAJ="${count}" \
      DRY_RUN=$((1 - EXECUTE)) bash "${SCRIPT_DIR}/collect_objectcentric_v3_sim2real.sh"
  done
done

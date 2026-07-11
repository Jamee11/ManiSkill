#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${MANISKILL_EORT_PYTHON:-/remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1/bin/python}
DATA_ROOT=${MANISKILL_EORT_DATA_ROOT:-/remote-home/jinminghao/datasets/maniskill_push_cube_objectcentric_v2}
RAW_ROOT="${DATA_ROOT}/raw"
OUTPUT_DIR="${DATA_ROOT}/derived"
NUM_TRAJ=${NUM_TRAJ:-10}
TRAJ_PATH="${RAW_ROOT}/PushCubeEORT-v1/motionplanning/push_cube_objectcentric_v2.h5"

if [[ -n "${MANISKILL_EORT_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${MANISKILL_EORT_CUDA_VISIBLE_DEVICES}"
fi

if [[ -e "${TRAJ_PATH}" || -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing EORT output under ${DATA_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" -m mani_skill.examples.motionplanning.panda.run \
  --env-id PushCubeEORT-v1 \
  --num-traj "${NUM_TRAJ}" \
  --only-count-success \
  --obs-mode state_dict+rgb+depth+segmentation \
  --sim-backend cpu \
  --record-dir "${RAW_ROOT}" \
  --traj-name push_cube_objectcentric_v2
"${PYTHON}" scripts/eort/derive_push_cube_eort.py \
  --traj-path "${TRAJ_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --schema objectcentric_v2

#!/usr/bin/env bash
set -euo pipefail

# Build one non-overwriting v3 shard:
# three-view raw RGB-D -> base-frame oracle sidecar -> tiled QA -> DiT4DiT LeRobot.
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DIT4DIT_ROOT=${DIT4DIT_ROOT:-/remote-home/jinminghao/DiT4DiT}
PYTHON=${MANISKILL_EORT_PYTHON:-/remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1/bin/python}
DIT4DIT_PYTHON=${DIT4DIT_PYTHON:-/remote-home/jinminghao/miniconda3/envs/cosmos-libero/bin/python}
COLLECTION_ROOT=${MANISKILL_EORT_COLLECTION_ROOT:-/remote-home/jinminghao/datasets/maniskill_eort_large_v3_sim2real}
TASK=${MANISKILL_EORT_TASK:-push_cube}
SPLIT=${MANISKILL_EORT_SPLIT:-train}
FPS=${MANISKILL_EORT_FPS:-20}
PREVIEW_EPISODES=${MANISKILL_EORT_PREVIEW_EPISODES:-3}
CONTROLLER_REPLAY_ENVS=${MANISKILL_EORT_CONTROLLER_REPLAY_ENVS:-1}
EXPORTS=${MANISKILL_EORT_EXPORTS:-oracle}

case "${TASK}" in
  push_cube) env_id=PushCubeEORTSim2Real-v1; dataset_prefix=maniskill_eort_push_cube_256_sim2real_robot_base ;;
  pick_cube) env_id=PickCubeEORTSim2Real-v1; dataset_prefix=maniskill_eort_pick_cube_256_sim2real_robot_base ;;
  *) echo "MANISKILL_EORT_TASK must be push_cube or pick_cube" >&2; exit 2 ;;
esac
case "${SPLIT}" in
  train) default_seed=0; default_count=500 ;;
  val) default_seed=1000000; default_count=100 ;;
  test) default_seed=2000000; default_count=200 ;;
  *) echo "MANISKILL_EORT_SPLIT must be train, val, or test" >&2; exit 2 ;;
esac
NUM_TRAJ=${NUM_TRAJ:-${default_count}}
START_SEED=${MANISKILL_EORT_START_SEED:-${default_seed}}
MAX_ATTEMPTS=${MANISKILL_EORT_MAX_ATTEMPTS:-$((NUM_TRAJ * 5))}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '{"schema":"objectcentric_v3","root":"%s","task":"%s","split":"%s","env":"%s","episodes":%s,"start_seed":%s,"exports":"%s","cameras":["front_camera","right_shoulder_camera","hand_camera"],"resolution":256,"frame":"robot_base","artificial_occluder":false}\n' \
    "${COLLECTION_ROOT}" "${TASK}" "${SPLIT}" "${env_id}" "${NUM_TRAJ}" "${START_SEED}" "${EXPORTS}"
  exit 0
fi
[[ -n "${MANISKILL_EORT_CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Set MANISKILL_EORT_CUDA_VISIBLE_DEVICES explicitly" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=${MANISKILL_EORT_CUDA_VISIBLE_DEVICES}
[[ -x "${PYTHON}" && -x "${DIT4DIT_PYTHON}" ]] || { echo "Missing configured Python executable" >&2; exit 1; }

shard_root=${COLLECTION_ROOT}/${TASK}/${SPLIT}/sim2real
raw_root=${shard_root}/raw
trajectory_name=${TASK}_${SPLIT}_sim2real_seed${START_SEED}_n${NUM_TRAJ}
trajectory_path=${raw_root}/${env_id}/motionplanning/${trajectory_name}.h5
source_seed_manifest=${trajectory_path%.h5}.seed_manifest.json
controller_path=${trajectory_path%.h5}.state_dict+rgb+depth+segmentation.pd_ee_delta_pose.physx_cpu.h5
controller_seed_manifest=${controller_path%.h5}.seed_manifest.json
derived_dir=${shard_root}/derived_objectcentric_v3
preview_path=${shard_root}/qa/${trajectory_name}_three_view.mp4
runtime_manifest=${shard_root}/runtime_manifest.json
for path in "${trajectory_path}" "${source_seed_manifest}" "${controller_path}" "${controller_seed_manifest}" "${derived_dir}" "${preview_path}" "${runtime_manifest}"; do
  [[ ! -e "${path}" ]] || { echo "Refusing to overwrite ${path}" >&2; exit 1; }
done

cd "${ROOT_DIR}"
"${PYTHON}" -m mani_skill.examples.motionplanning.panda.run \
  --env-id "${env_id}" --num-traj "${NUM_TRAJ}" --only-count-success \
  --start-seed "${START_SEED}" --max-attempts "${MAX_ATTEMPTS}" --save-seed-manifest \
  --obs-mode state_dict+rgb+depth+segmentation --sim-backend cpu \
  --record-dir "${raw_root}" --traj-name "${trajectory_name}"
"${PYTHON}" -m mani_skill.trajectory.replay_trajectory \
  --traj-path "${trajectory_path}" --use-first-env-state \
  --target-control-mode pd_ee_delta_pose --obs-mode state_dict+rgb+depth+segmentation \
  --save-traj --num-envs "${CONTROLLER_REPLAY_ENVS}" --sim-backend physx_cpu
[[ -f "${controller_path}" ]] || { echo "Missing controller replay ${controller_path}" >&2; exit 1; }
cp "${source_seed_manifest}" "${controller_seed_manifest}"

"${PYTHON}" scripts/eort/derive_push_cube_eort.py \
  --traj-path "${controller_path}" --output-dir "${derived_dir}" --schema objectcentric_v3 \
  --cameras front_camera right_shoulder_camera hand_camera
"${PYTHON}" scripts/eort/render_objectcentric_preview.py \
  --trajectory-path "${controller_path}" --derived-dir "${derived_dir}" --output "${preview_path}" \
  --cameras front_camera right_shoulder_camera hand_camera --max-episodes "${PREVIEW_EPISODES}" --fps "${FPS}"

IFS=',' read -r -a exports <<< "${EXPORTS}"
for export_name in "${exports[@]}"; do
  export_args=()
  case "${export_name}" in
    oracle) condition=oracle ;;
    learned_tracker)
      condition=learned_tracker
      [[ -f "${MANISKILL_EORT_TRACKER_CHECKPOINT:-}" ]] || { echo "Set MANISKILL_EORT_TRACKER_CHECKPOINT for learned_tracker export" >&2; exit 2; }
      export_args=(--tracker-checkpoint "${MANISKILL_EORT_TRACKER_CHECKPOINT}" --tracker-device "${MANISKILL_EORT_TRACKER_DEVICE:-cpu}")
      ;;
    *) echo "v3 exports are oracle or learned_tracker, got ${export_name}" >&2; exit 2 ;;
  esac
  "${DIT4DIT_PYTHON}" "${DIT4DIT_ROOT}/examples/RLBench_EORT/scripts/convert_maniskill_eort_to_lerobot.py" \
    --trajectory-path "${controller_path}" --derived-dir "${derived_dir}" \
    --output-root "${COLLECTION_ROOT}/lerobot/${SPLIT}/${condition}" \
    --dataset-name "${dataset_prefix}_${condition}_lerobot" --fps "${FPS}" \
    --cameras front_camera right_shoulder_camera hand_camera \
    --condition-source "${condition}" --action-source metric_robot_base_delta_pose \
    "${export_args[@]}"
done

"${PYTHON}" - "${runtime_manifest}" "${CUDA_VISIBLE_DEVICES}" <<'PY'
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "maniskill_eort_runtime_v3",
    "python": platform.python_version(),
    "packages": {name: metadata.version(name) for name in ("mani_skill", "torch", "sapien", "numpy", "h5py")},
    "cuda_visible_devices": sys.argv[2],
    "cameras": {
        "front_camera": {"serial": "344522302193", "calibration": "2026-06-17 real matched", "resolution": [256, 256]},
        "right_shoulder_camera": {"serial": "344422300343", "calibration": "2026-01-16 real matched candidate", "resolution": [256, 256]},
        "hand_camera": {"serial": "339222070579", "calibration": "simulator nominal; real intrinsics/hand-eye unresolved", "resolution": [256, 256]},
    },
    "real_preprocess": "undistort 1280x720, center-crop 720x720, resize 256x256",
    "canonical_frame": "robot_base",
    "artificial_occluder": False,
}, indent=2, sort_keys=True) + "\n")
PY

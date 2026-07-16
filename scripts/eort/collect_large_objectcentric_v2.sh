#!/usr/bin/env bash
set -euo pipefail

# One command builds a disjoint ManiSkill EORT shard: raw HDF5 -> sidecar ->
# overlay preview -> LeRobot oracle/proxy/corruption views. It never overwrites.
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DIT4DIT_ROOT=${DIT4DIT_ROOT:-/remote-home/jinminghao/DiT4DiT}
PYTHON=${MANISKILL_EORT_PYTHON:-/remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1/bin/python}
DIT4DIT_PYTHON=${DIT4DIT_PYTHON:-/remote-home/jinminghao/miniconda3/envs/cosmos-libero/bin/python}
COLLECTION_ROOT=${MANISKILL_EORT_COLLECTION_ROOT:-/remote-home/jinminghao/datasets/maniskill_eort_large_v2}
TASK=${MANISKILL_EORT_TASK:-push_cube}
SPLIT=${MANISKILL_EORT_SPLIT:-train}
VARIANTS=${MANISKILL_EORT_VARIANTS:-fixed,camera_rand,occluded}
NUM_TRAJ=${NUM_TRAJ:-1000}
SEED_BLOCK_SIZE=${MANISKILL_EORT_SEED_BLOCK_SIZE:-100000}
PREVIEW_EPISODES=${MANISKILL_EORT_PREVIEW_EPISODES:-3}
CONTROLLER_REPLAY_ENVS=${MANISKILL_EORT_CONTROLLER_REPLAY_ENVS:-1}
FPS=${MANISKILL_EORT_FPS:-20}
EXPORTS=${MANISKILL_EORT_EXPORTS:-oracle,segdepth_proxy,track_corrupt}
TRACK_DELAY=${MANISKILL_EORT_TRACK_DELAY:-2}
TRACK_ID_SWITCH_PROB=${MANISKILL_EORT_TRACK_ID_SWITCH_PROB:-0.3}
TRACK_CORRUPTION_SEED=${MANISKILL_EORT_TRACK_CORRUPTION_SEED:-17}
ACTION_SOURCE=${MANISKILL_EORT_ACTION_SOURCE:-panda_pd_ee_delta_pose}

case "${ACTION_SOURCE}" in
  panda_pd_ee_delta_pose) action_output_suffix= ;;
  metric_task_delta_pose) action_output_suffix=_metric ;;
  *) echo "MANISKILL_EORT_ACTION_SOURCE must be panda_pd_ee_delta_pose or metric_task_delta_pose" >&2; exit 2 ;;
esac

case "${TASK}" in
  push_cube) dataset_prefix=maniskill_eort_push_cube_256 ;;
  pick_cube) dataset_prefix=maniskill_eort_pick_cube_256 ;;
  *) echo "MANISKILL_EORT_TASK must be push_cube or pick_cube" >&2; exit 2 ;;
esac
case "${SPLIT}" in
  train) split_seed_default=0 ;;
  val) split_seed_default=1000000 ;;
  test) split_seed_default=2000000 ;;
  *) echo "MANISKILL_EORT_SPLIT must be train, val, or test" >&2; exit 2 ;;
esac
START_SEED=${MANISKILL_EORT_START_SEED:-${split_seed_default}}
if (( NUM_TRAJ <= 0 || NUM_TRAJ >= SEED_BLOCK_SIZE )); then
  echo "NUM_TRAJ must be positive and smaller than MANISKILL_EORT_SEED_BLOCK_SIZE" >&2
  exit 2
fi
if [[ -n "${MANISKILL_EORT_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${MANISKILL_EORT_CUDA_VISIBLE_DEVICES}"
fi

variant_env() {
  case "$1" in
    fixed) echo "${2}EORT-v1" ;;
    camera_rand) echo "${2}EORTCameraRand-v1" ;;
    occluded) echo "${2}EORTOccluded-v1" ;;
    *) echo "unknown variant: $1" >&2; return 2 ;;
  esac
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '{"collection_root":"%s","task":"%s","split":"%s","variants":"%s","num_traj_per_variant":%s,"start_seed":%s,"seed_block_size":%s,"exports":"%s","action_source":"%s","controller_replay_envs":%s,"gpu":"%s"}\n' \
    "${COLLECTION_ROOT}" "${TASK}" "${SPLIT}" "${VARIANTS}" "${NUM_TRAJ}" "${START_SEED}" "${SEED_BLOCK_SIZE}" "${EXPORTS}" "${ACTION_SOURCE}" "${CONTROLLER_REPLAY_ENVS}" "${CUDA_VISIBLE_DEVICES:-unset}"
  exit 0
fi
if [[ ! -x "${PYTHON}" || ! -x "${DIT4DIT_PYTHON}" || ! -f "${DIT4DIT_ROOT}/examples/RLBench_EORT/scripts/convert_maniskill_eort_to_lerobot.py" ]]; then
  echo "Missing ManiSkill/DiT4DiT Python or exporter; set MANISKILL_EORT_PYTHON, DIT4DIT_PYTHON, DIT4DIT_ROOT" >&2
  exit 1
fi

IFS=',' read -r -a variants <<< "${VARIANTS}"
IFS=',' read -r -a exports <<< "${EXPORTS}"
for index in "${!variants[@]}"; do
  variant=${variants[$index]}
  base_env=$([[ "${TASK}" == push_cube ]] && echo PushCube || echo PickCube)
  env_id=$(variant_env "${variant}" "${base_env}")
  seed=$((START_SEED + index * SEED_BLOCK_SIZE))
  shard_root="${COLLECTION_ROOT}/${TASK}/${SPLIT}/${variant}"
  raw_root="${shard_root}/raw"
  trajectory_name="${TASK}_${SPLIT}_${variant}_seed${seed}_n${NUM_TRAJ}"
  trajectory_path="${raw_root}/${env_id}/motionplanning/${trajectory_name}.h5"
  source_seed_manifest="${trajectory_path%.h5}.seed_manifest.json"
  controller_path="${trajectory_path%.h5}.state_dict+rgb+depth+segmentation.pd_ee_delta_pose.physx_cpu.h5"
  controller_seed_manifest="${controller_path%.h5}.seed_manifest.json"
  derived_dir="${shard_root}/derived_controller_goal_segdepth"
  preview_path="${shard_root}/qa/${trajectory_name}_preview.mp4"
  if [[ -e "${trajectory_path}" || -e "${source_seed_manifest}" || -e "${controller_path}" || -e "${controller_seed_manifest}" || -e "${derived_dir}" || -e "${preview_path}" ]]; then
    echo "Refusing to overwrite shard output under ${shard_root}" >&2
    exit 1
  fi
  cd "${PROJECT_ROOT}"
  "${PYTHON}" -m mani_skill.examples.motionplanning.panda.run \
    --env-id "${env_id}" --num-traj "${NUM_TRAJ}" --only-count-success \
    --start-seed "${seed}" --save-seed-manifest \
    --obs-mode state_dict+rgb+depth+segmentation --sim-backend cpu \
    --record-dir "${raw_root}" --traj-name "${trajectory_name}"
  "${PYTHON}" -m mani_skill.trajectory.replay_trajectory \
    --traj-path "${trajectory_path}" --use-first-env-state \
    --target-control-mode pd_ee_delta_pose --obs-mode state_dict+rgb+depth+segmentation \
    --save-traj --num-envs "${CONTROLLER_REPLAY_ENVS}" --sim-backend physx_cpu
  [[ -f "${controller_path}" ]] || { echo "Missing controller replay output: ${controller_path}" >&2; exit 1; }
  controller_count=$("${PYTHON}" -c 'import h5py,sys; f=h5py.File(sys.argv[1]); print(len(f)); f.close()' "${controller_path}")
  if (( controller_count != NUM_TRAJ )); then
    echo "Controller replay kept ${controller_count}/${NUM_TRAJ} trajectories; refusing partial shard" >&2
    exit 1
  fi
  cp "${source_seed_manifest}" "${controller_seed_manifest}"
  "${PYTHON}" scripts/eort/derive_push_cube_eort.py \
    --traj-path "${controller_path}" --output-dir "${derived_dir}" --schema objectcentric_v2
  "${PYTHON}" scripts/eort/render_objectcentric_preview.py \
    --trajectory-path "${controller_path}" --derived-dir "${derived_dir}" \
    --output "${preview_path}" --max-episodes "${PREVIEW_EPISODES}" --fps "${FPS}"
  for export_name in "${exports[@]}"; do
    case "${export_name}" in
      oracle)
        output_root="${COLLECTION_ROOT}/lerobot/${SPLIT}/oracle${action_output_suffix}"
        dataset_name="${dataset_prefix}_${variant}_lerobot"
        export_args=(--condition-source oracle)
        ;;
      segdepth_proxy)
        output_root="${COLLECTION_ROOT}/lerobot/${SPLIT}/segdepth_proxy${action_output_suffix}"
        dataset_name="${dataset_prefix}_${variant}_segdepth_track_proxy_lerobot"
        export_args=(--condition-source segdepth_track_proxy)
        ;;
      track_corrupt)
        output_root="${COLLECTION_ROOT}/lerobot/${SPLIT}/track_corrupt${action_output_suffix}"
        dataset_name="${dataset_prefix}_${variant}_segdepth_track_corrupt_lerobot"
        export_args=(--condition-source segdepth_track_proxy --track-delay-steps "${TRACK_DELAY}" --track-id-switch-prob "${TRACK_ID_SWITCH_PROB}" --track-corruption-seed "${TRACK_CORRUPTION_SEED}")
        ;;
      *) echo "Unknown MANISKILL_EORT_EXPORTS item: ${export_name}" >&2; exit 2 ;;
    esac
    "${DIT4DIT_PYTHON}" "${DIT4DIT_ROOT}/examples/RLBench_EORT/scripts/convert_maniskill_eort_to_lerobot.py" \
      --trajectory-path "${controller_path}" --derived-dir "${derived_dir}" \
      --output-root "${output_root}" --dataset-name "${dataset_name}" --fps "${FPS}" \
      --action-source "${ACTION_SOURCE}" "${export_args[@]}"
  done
done

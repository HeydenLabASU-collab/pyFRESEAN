#!/usr/bin/env bash
# Submit full HEWL benchmark sweep (CG once, C spectral, py FRESEAN cases).
#
# Usage:
#   bash bench/hewl/submit.sh           # submit everything
#   bash bench/hewl/submit.sh cg        # py CG reference only
#   bash bench/hewl/submit.sh c         # C spectral sweep only
#   bash bench/hewl/submit.sh py        # py FRESEAN case sweeps only
#
set -euo pipefail

PHASE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYFRESEAN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS="${SCRIPT_DIR}/results"
CPUS=(1 2 4 8 16 32 48)
STAGGER_SEC=20
CASES=(
  omp1_njobs_n_nworkers_1
  omp1_njobs_n_nworkers_2
  omp1_njobs_n_nworkers_n
  omp_n_njobs_1
)

_bench() {
  echo "cd ${PYFRESEAN_ROOT} && python3 bench/hewl/benchmark.py $*"
}

_submit() {
  local name="$1"
  local n="$2"
  local out_dir="$3"
  local cmd="$4"
  local cpus="${5:-$n}"
  local dep="${6:-}"
  mkdir -p "${out_dir}/logs"
  local dep_flag=()
  if [[ -n "${dep}" ]]; then
    dep_flag=(--dependency="afterok:${dep}")
  fi
  sbatch \
    "${dep_flag[@]}" \
    --chdir="${PYFRESEAN_ROOT}" \
    --partition=public \
    --time=08:00:00 \
    --job-name="${name}" \
    --cpus-per-task="${cpus}" \
    --export=ALL,NCPUS="${cpus}",PYFRESEAN_ROOT="${PYFRESEAN_ROOT}" \
    --output="${out_dir}/logs/slurm_%x_%j.out" \
    --error="${out_dir}/logs/slurm_%x_%j.err" \
    --wrap="${cmd}"
}

_submit_cg() {
  local out="${SCRIPT_DIR}/cg_reference/pyfresean"
  _submit "hewl-cg-ref" 1 "${out}" \
    "$(_bench --mode cg --ncpus 1 --output-dir bench/hewl/cg_reference/pyfresean)" 1
}

_submit_c() {
  local out="${RESULTS}/c_spectral"
  for n in "${CPUS[@]}"; do
    _submit "hewl-c-${n}" "${n}" "${out}" \
        "$(_bench --mode c --ncpus ${n} --skip-c-inputs-check --c-inputs-dir bench/hewl/cg_reference/c_ref --output-dir bench/hewl/results/c_spectral)" \
      "${n}"
    sleep "${STAGGER_SEC}"
  done
}

_submit_py_cases() {
  local dep="${1:-}"
  for case in "${CASES[@]}"; do
    local out="${RESULTS}/py_cases/${case}"
    for n in "${CPUS[@]}"; do
      _submit "hewl-py-${case}-${n}" "${n}" "${out}" \
        "$(_bench --mode fresean --case ${case} --ncpus ${n} --skip-c-inputs-check --cg-cache-dir bench/hewl/cg_reference/pyfresean --cg-reference-dir bench/hewl/cg_reference/pyfresean --output-dir bench/hewl/results/py_cases/${case})" \
        "${n}" "${dep}"
      sleep "${STAGGER_SEC}"
    done
  done
}

case "${PHASE}" in
  all)
    CG_JOB="$(_submit_cg | awk '{print $4}')"
    echo "CG reference job: ${CG_JOB}"
    sleep "${STAGGER_SEC}"
    _submit_c
    sleep "${STAGGER_SEC}"
    _submit_py_cases "${CG_JOB}"
    ;;
  cg) _submit_cg ;;
  c) _submit_c ;;
  py) _submit_py_cases ;;
  *)
    echo "usage: submit.sh [all|cg|c|py]" >&2
    exit 2
    ;;
esac

echo "Submitted phase=${PHASE}."
echo "After jobs finish: python bench/hewl/collect_results.py --case all --plot"

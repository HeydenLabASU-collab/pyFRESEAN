#!/usr/bin/env bash
# Submit HEWL benchmark sweeps.
#
# Usage:
#   bash bench/hewl/submit.sh                  # full CG sweep (all cases)
#   bash bench/hewl/submit.sh py_cg            # py CG once
#   bash bench/hewl/submit.sh c_cg             # C CG once
#   bash bench/hewl/submit.sh c_aa             # C AA inputs once
#   bash bench/hewl/submit.sh py_fresean       # py FRESEAN CG case sweeps
#   bash bench/hewl/submit.sh py_fresean_vec   # vectorized vec cases only
#   bash bench/hewl/submit.sh py_fresean_vec_omp1  # OMP=1, n_jobs=N vec test only
#   bash bench/hewl/submit.sh py_fresean_hybrid   # hybrid: corr n_jobs=N, rfft/eigh OMP=N
#   bash bench/hewl/submit.sh c_spectral       # C spectral CG sweep
#   bash bench/hewl/submit.sh aa               # AA py+C spectral (omp_n_njobs_1)
#
# Legacy phase names: cg → py_cg, py → py_fresean, c → c_spectral
set -euo pipefail

PHASE="${1:-all}"
case "${PHASE}" in
  cg) PHASE="py_cg" ;;
  py) PHASE="py_fresean" ;;
  c) PHASE="c_spectral" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYFRESEAN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_CG="${SCRIPT_DIR}/results/results_cg"
RESULTS_AA="${SCRIPT_DIR}/results/results_aa"
CPUS=(1 2 4 8 16 32 48)
STAGGER_SEC=20
CASES=(
  omp1_njobs_n_nworkers_1
  omp1_njobs_n_nworkers_2
  omp1_njobs_n_nworkers_n
  omp_n_njobs_1
  omp_n_njobs_1_vec
  omp1_njobs_n_vec
)
AA_CASE="omp_n_njobs_1"
VEC_CASES=(
  omp_n_njobs_1_vec
  omp1_njobs_n_vec
)
VEC_OMP1_CASES=(
  omp1_njobs_n_vec
)
HYBRID_CASES=(
  omp_hybrid_vec
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

_submit_py_cg() {
  local out="${SCRIPT_DIR}/cg_reference/pyfresean"
  _submit "hewl-py-cg" 1 "${out}" \
    "$(_bench --mode py_cg --ncpus 1 --output-dir bench/hewl/cg_reference/pyfresean)" 1
}

_submit_c_cg() {
  local out="${SCRIPT_DIR}/cg_reference/c_ref"
  _submit "hewl-c-cg" 1 "${out}" \
    "$(_bench --mode c_cg --ncpus 1 --output-dir bench/hewl/cg_reference/c_ref)" 1
}

_submit_c_aa() {
  local out="${SCRIPT_DIR}/aa_reference/c_ref"
  _submit "hewl-c-aa" 1 "${out}" \
    "$(_bench --mode c_aa --system aa --ncpus 1 --output-dir bench/hewl/aa_reference/c_ref)" 1
}

_submit_c_spectral() {
  local dep="${1:-}"
  local system="${2:-cg}"
  local results_var="RESULTS_CG"
  if [[ "${system}" == "aa" ]]; then
    results_var="RESULTS_AA"
  fi
  local out="${!results_var}/c_spectral"
  for n in "${CPUS[@]}"; do
    _submit "hewl-c-${system}-spectral-${n}" "${n}" "${out}" \
      "$(_bench --mode c_spectral --system ${system} --ncpus ${n} --skip-c-inputs-check --c-inputs-dir bench/hewl/${system}_reference/c_ref --output-dir bench/hewl/results/results_${system}/c_spectral)" \
      "${n}" "${dep}"
    sleep "${STAGGER_SEC}"
  done
}

_submit_py_fresean_cases() {
  local dep="${1:-}"
  local system="${2:-cg}"
  shift 2
  local case_list=("$@")
  local results_var="RESULTS_CG"
  if [[ "${system}" == "aa" ]]; then
    results_var="RESULTS_AA"
  fi
  for case in "${case_list[@]}"; do
    local out="${!results_var}/py_cases/${case}"
    for n in "${CPUS[@]}"; do
      local extra_args=()
      if [[ "${system}" == "cg" ]]; then
        extra_args=(
          --skip-c-inputs-check
          --cg-cache-dir bench/hewl/cg_reference/pyfresean
          --cg-reference-dir bench/hewl/cg_reference/pyfresean
        )
      fi
      _submit "hewl-py-${system}-${case}-${n}" "${n}" "${out}" \
        "$(_bench --mode py_fresean --system ${system} --case ${case} --ncpus ${n} ${extra_args[*]} --output-dir bench/hewl/results/results_${system}/py_cases/${case})" \
        "${n}" "${dep}"
      sleep "${STAGGER_SEC}"
    done
  done
}

_submit_py_fresean() {
  local dep="${1:-}"
  local system="${2:-cg}"
  if [[ "${system}" == "aa" ]]; then
    _submit_py_fresean_cases "${dep}" "${system}" "${AA_CASE}"
  else
    _submit_py_fresean_cases "${dep}" "${system}" "${CASES[@]}"
  fi
}

_submit_aa() {
  C_AA_JOB="$(_submit_c_aa | awk '{print $4}')"
  echo "C AA inputs job: ${C_AA_JOB}"
  sleep "${STAGGER_SEC}"
  _submit_c_spectral "${C_AA_JOB}" "aa"
  sleep "${STAGGER_SEC}"
  _submit_py_fresean "" "aa"
}

case "${PHASE}" in
  all)
    PY_CG_JOB="$(_submit_py_cg | awk '{print $4}')"
    C_CG_JOB="$(_submit_c_cg | awk '{print $4}')"
    echo "py CG job: ${PY_CG_JOB}"
    echo "C CG job: ${C_CG_JOB}"
    sleep "${STAGGER_SEC}"
    _submit_c_spectral "${C_CG_JOB}" "cg"
    sleep "${STAGGER_SEC}"
    _submit_py_fresean "${PY_CG_JOB}" "cg"
    ;;
  py_cg) _submit_py_cg ;;
  c_cg) _submit_c_cg ;;
  c_aa) _submit_c_aa ;;
  c_spectral) _submit_c_spectral "" "cg" ;;
  py_fresean) _submit_py_fresean "" "cg" ;;
  py_fresean_vec)
    _submit_py_fresean_cases "" "cg" "${VEC_CASES[@]}"
    ;;
  py_fresean_vec_omp1)
    _submit_py_fresean_cases "" "cg" "${VEC_OMP1_CASES[@]}"
    ;;
  py_fresean_hybrid)
    _submit_py_fresean_cases "" "cg" "${HYBRID_CASES[@]}"
    ;;
  aa) _submit_aa ;;
  *)
    echo "usage: submit.sh [all|py_cg|c_cg|c_aa|py_fresean|py_fresean_vec|py_fresean_vec_omp1|py_fresean_hybrid|c_spectral|aa]" >&2
    echo "       legacy: cg, py, c" >&2
    exit 2
    ;;
esac

echo "Submitted phase=${PHASE}."
echo "After jobs finish: python bench/hewl/collect_results.py --system all --case all --plot"

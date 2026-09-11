#!/usr/bin/env bash
# One-time: build C covar/eigen inputs (topol-cg.mtop, ref-cg.gro, traj-cg.trr link).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYFRESEAN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HEWL_DATA="${HEWL_DATA_DIR:-${PYFRESEAN_ROOT}/pyfresean/tests/data/fresean_c_ref/hewl_solution_303K}"
INPUTS_DIR="${SCRIPT_DIR}/cg_reference/c_ref"
C_REF_SCRIPT="${PYFRESEAN_ROOT}/devtools/scripts/fresean_c_ref/hewl_solution_303K"

TPR="${HEWL_DATA}/output/topol_prot.tpr"
TRJ="${HEWL_DATA}/output/sample-NPT_prot_pbc.trr"
TOP_PROT="${HEWL_DATA}/input/topol_prot.top"
FRESEAN_BIN="${FRESEAN_BIN:-fresean}"
GMX="${GMX:-gmx}"
N_FRAMES=5000
DT=0.01

module load gromacs 2>/dev/null || true

for f in "${TPR}" "${TRJ}" "${TOP_PROT}"; do
  [[ -f "${f}" ]] || { echo "Missing: ${f}"; exit 1; }
done

if [[ -f "${INPUTS_DIR}/topol-cg.mtop" && -f "${INPUTS_DIR}/ref-cg.gro" ]]; then
  echo "C covar inputs already present: ${INPUTS_DIR}"
  exit 0
fi

rm -rf "${INPUTS_DIR}"
mkdir -p "${INPUTS_DIR}"
cd "${INPUTS_DIR}"

cp "${C_REF_SCRIPT}/static.job" .

printf '%s\n%s\n%s\n' "${TOP_PROT}" "${TOP_PROT}" "topol-aa.mtop" | \
  "${FRESEAN_BIN}" mtop -p "${TOP_PROT}"

TRJ_END=$(python3 -c "print(${N_FRAMES} * ${DT})")
echo 0 | "${GMX}" trjconv -s "${TPR}" -f "${TRJ}" -o aa.trr -b 0 -e "${TRJ_END}" 2>&1 | tail -3

cat > coarse.inp <<EOF
#fnTop
topol-aa.mtop
#fnCrd
aa.trr
#fnVel
#fnJob
static.job
#grp
0
#nRead
${N_FRAMES}
#nSample
1
#fnOutTraj
tmptraj.gro
#fnOutTopol
topol-cg.mtop
EOF
"${FRESEAN_BIN}" coarse -f coarse.inp

N_ATOMS=$(head -n 2 tmptraj.gro | tail -n 1)
N_LINES=$((N_ATOMS + 3))
head -n "${N_LINES}" tmptraj.gro > ref-cg.gro
rm -f tmptraj.gro aa.trr topol-aa.mtop coarse.inp
ln -sf "${HEWL_DATA}/traj-cg.trr" traj-cg.trr

echo "Ready: ${INPUTS_DIR}"

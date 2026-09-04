#!/usr/bin/env bash
# Run FRESEAN COARSE (C) and write stored reference files for c_ref tests.
#
# Prerequisites: fresean (FRESEANCOARSE), gmx, GSL for fresean eigen (see README)
# Input: examples/input_data/MD-gas-300K/
# Output: PYFRESEAN_C_REF_DIR (only traj-cg.trr, eval_covar_cg.mmat.dat, evec_covar_cg.mmat)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYFRESEAN_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${PYFRESEAN_C_REF_DIR:-${PYFRESEAN_ROOT}/pyfresean/tests/data/fresean_c_ref/ala_dipeptide_gas_300K}"
DATA_ROOT="${PYFRESEAN_TEST_DATA:-${PYFRESEAN_ROOT}/examples/input_data}"
STAGING_DIR="${SCRIPT_DIR}/output/staging"

AA_DIR="${DATA_ROOT}/MD-gas-300K"
TPR="${AA_DIR}/topol.tpr"
TRJ="${AA_DIR}/traj.trr"
HARMONIC="${DATA_ROOT}/harmonic-normal-modes/min.xyz"
TOP_PROT="${SCRIPT_DIR}/topol_prot.top"

FRESEAN_BIN="${FRESEAN_BIN:-fresean}"
GMX="${GMX:-gmx}"

N_FRAMES=5000
N_CORR=100
DT=0.004
WIN_SIGMA=10.0

REFERENCE_FILES=(
  "traj-cg.trr"
  "eval_covar_cg.mmat.dat"
  "evec_covar_cg.mmat"
)

for f in "${TPR}" "${TRJ}" "${HARMONIC}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing required file: ${f}"
    exit 1
  fi
done

if [[ ! -f "${TOP_PROT}" ]]; then
  echo "Missing ${TOP_PROT}. Run generate_topol_prot.sh."
  exit 1
fi

# fresean eigen is a separate binary that links libgsl
EIGEN_BIN="${EIGEN_BIN:-$(command -v eigen 2>/dev/null || true)}"
if [[ -n "${EIGEN_BIN}" ]] && ldd "${EIGEN_BIN}" 2>/dev/null | grep -q 'libgsl.so.*not found'; then
  echo "eigen cannot load GSL (${EIGEN_BIN})."
  echo "  module load gsl"
  echo "  cd FRESEANCOARSE && make eigen"
  exit 1
fi

rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}" "${OUTPUT_DIR}"
cd "${STAGING_DIR}"

cp "${SCRIPT_DIR}/static.job" .

echo "=== fresean mtop (all-atom) ==="
printf '%s\n%s\n%s\n' "${TOP_PROT}" "${TOP_PROT}" "topol-aa.mtop" | \
  "${FRESEAN_BIN}" mtop -p "${TOP_PROT}"

TRJ_END=$(python3 -c "print(${N_FRAMES} * ${DT})")

echo "=== gmx: protein trajectory (first ${N_FRAMES} frames) ==="
echo 0 | "${GMX}" trjconv -s "${TPR}" -f "${TRJ}" -o aa.trr -b 0 -e "${TRJ_END}" 2>&1 | tail -3

echo "=== fresean coarse ==="
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

echo "=== gmx: CG trajectory TRR ==="
echo 0 | "${GMX}" trjconv -s tmptraj.gro -f tmptraj.gro -timestep "${DT}" -o traj-cg.trr 2>&1 | tail -3

N_ATOMS=$(head -n 2 tmptraj.gro | tail -n 1)
N_LINES=$((N_ATOMS + 3))
head -n "${N_LINES}" tmptraj.gro > ref-cg.gro
rm -f tmptraj.gro

echo "=== fresean covar + eigen ==="
cat > covar.inp <<EOF
#fnTop
topol-cg.mtop
#fnCrd
traj-cg.trr
#fnJob
static.job
#nRead
${N_FRAMES}
#analysisInterval
1
#fnRef
ref-cg.gro
#alignGrp
0
#analyzeGrp
0
#wrap
0
#nCorr
${N_CORR}
#winSigma
${WIN_SIGMA}
#binaryMatrix
1
#doGenModes
0
#convergence
1.0e-5
#maxIter
100
#fnOut
cg
EOF

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
"${FRESEAN_BIN}" covar -f covar.inp
"${FRESEAN_BIN}" eigen -m covar_cg.mmat -n "${N_CORR}"

for name in "${REFERENCE_FILES[@]}"; do
  cp "${STAGING_DIR}/${name}" "${OUTPUT_DIR}/${name}"
done

rm -rf "${STAGING_DIR}"

echo "C reference written to ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}/traj-cg.trr" "${OUTPUT_DIR}/eval_covar_cg.mmat.dat" "${OUTPUT_DIR}/evec_covar_cg.mmat"

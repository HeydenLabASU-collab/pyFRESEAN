#!/usr/bin/env bash
# Flatten CHARMM-GUI PROA topology for fresean mtop (requires last line: Protein 1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYFRESEAN_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
HEWL_DATA="${HEWL_DATA_DIR:-${PYFRESEAN_ROOT}/pyfresean/tests/data/fresean_c_ref/hewl_solution_303K}"
TOPPAR="${HEWL_DATA}/input/toppar"
OUT_INPUT="${HEWL_DATA}/input/topol_prot.top"
OUT_SCRIPT="${SCRIPT_DIR}/topol_prot.top"

if [[ ! -f "${TOPPAR}/PROA.itp" ]]; then
  echo "Missing ${TOPPAR}/PROA.itp"
  exit 1
fi

{
  echo "; Standalone protein topology for fresean mtop (HEWL / PROA)"
  cat "${TOPPAR}/forcefield.itp"
  sed 's/^PROA[[:space:]]/Protein /' "${TOPPAR}/PROA.itp"
  cat <<EOF

[ system ]
HEWL protein

[ molecules ]
; Compound        #mols
Protein             1
EOF
} > "${OUT_SCRIPT}"

cp "${OUT_SCRIPT}" "${OUT_INPUT}"
echo "Written ${OUT_INPUT}"

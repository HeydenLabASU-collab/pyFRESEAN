#!/usr/bin/env bash
# Protein-only .top for fresean mtop. pdb2gmx ACE order differs from topol.tpr;
# swap atoms 1↔2 so topology matches traj.trr.
#
# Usage: bash generate_topol_prot.sh examples/input_data/MD-gas-300K/topol.tpr

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPR="${1:?Usage: $0 /path/to/MD-gas-300K/topol.tpr}"
GMX="${GMX:-gmx}"
FF="${FORCEFIELD:-amber99sb-ildn}"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT
cd "${WORKDIR}"

"${GMX}" editconf -f "${TPR}" -o conf.gro &>/dev/null
printf '1\n' | "${GMX}" pdb2gmx -f conf.gro -ff "${FF}" -water none -ignh -p topol.top &>/dev/null

python3 <<'PY'
from pathlib import Path

lines = Path("topol.top").read_text().splitlines()
section = None
atom_at = {}

for i, line in enumerate(lines):
    s = line.strip()
    if s.startswith("[") and s.endswith("]"):
        section = s[2:-1].strip().lower()
        continue
    if not s or s.startswith(";"):
        continue
    if section == "atoms" and s.split()[0].isdigit():
        atom_at[int(s.split()[0])] = i
    elif section in ("bonds", "pairs", "angles", "dihedrals"):
        lines[i] = " ".join(
            "2" if p == "1" else "1" if p == "2" else p for p in s.split()
        )

if 1 in atom_at and 2 in atom_at:
    a, b = lines[atom_at[1]].split(), lines[atom_at[2]].split()
    lines[atom_at[1]] = "1 " + " ".join(b[1:])
    lines[atom_at[2]] = "2 " + " ".join(a[1:])

Path("topol_prot.top").write_text("\n".join(lines) + "\n")
PY

cp topol_prot.top "${SCRIPT_DIR}/topol_prot.top"
echo "Written ${SCRIPT_DIR}/topol_prot.top"

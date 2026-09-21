#!/usr/bin/env bash
# Post-process output/run.{tpr,trr} -> protein-only files for FRESEAN / pyfresean.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
module load gromacs

PROTEIN_GRP=1

echo "convert-tpr -> output/topol_prot.tpr"
echo "${PROTEIN_GRP}" | gmx convert-tpr -s output/run.tpr -o output/topol_prot.tpr \
  >& output/convert_tpr.log

echo "trjconv -> output/sample-NPT_prot_pbc.trr"
echo "${PROTEIN_GRP}" | gmx trjconv -s output/run.tpr -f output/run.trr \
  -o output/sample-NPT_prot_pbc.trr -pbc mol -ur compact \
  >& output/trjconv.log

echo "Done:"
ls -lah output/topol_prot.tpr output/sample-NPT_prot_pbc.trr

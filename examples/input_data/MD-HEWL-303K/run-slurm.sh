#!/usr/bin/env bash
#SBATCH --job-name=hewl_303K_0.1ns
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Ref: https://github.com/amruthesht/imd-workshop-2025/tree/main/workshop/sample_simulation/GROMACS/input

set -euo pipefail

mkdir -p output

module load gromacs

if ! command -v gmx &>/dev/null; then
  echo "gmx not found in PATH."
  exit 1
fi

NTOMP="${NTOMP:-12}"

echo "[$(date -Iseconds)] Job on $(hostname), gmx=$(command -v gmx)"
gmx --version | head -3

echo "grompp..."
gmx grompp -f input/input.mdp \
  -c input/start.gro \
  -p input/topol.top \
  -n input/index.ndx \
  -o output/run.tpr \
  -maxwarn 1 >& output/grompp.log

echo "mdrun (0.1 ns)..."
gmx mdrun -deffnm output/run -v \
  -ntmpi 1 -ntomp "${NTOMP}" \
  -nb gpu -pme gpu -bonded gpu -update gpu \
  >& output/mdrun.log

echo "[$(date -Iseconds)] Done. Trajectory: output/run.trr"

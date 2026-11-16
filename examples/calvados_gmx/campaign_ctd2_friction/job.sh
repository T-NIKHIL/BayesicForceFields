#!/bin/bash
#SBATCH --job-name=ctd2_fric
#SBATCH --time=1:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL

# Friction-validation array: one task per Sobol vector re-run at the
# corrected friction 0.01 ps^-1 (the parent campaign ran at the mistaken
# 1.16 ps^-1). Uses run_ctd2_friction_campaign.py with its 10 ns
# equilibration + 100 ns production protocol.
#
# Submit from the login node:
#   cd ~/BayesicForceFields/examples/calvados_gmx
#   sbatch --array=1-3%3 campaign_ctd2_friction/job.sh
# (task id i -> sample index i-1; vec_000..vec_002 by default)

set -euo pipefail

REPO="$HOME/BayesicForceFields"
SCRIPT_DIR="$REPO/examples/calvados_gmx"
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
SAMPLE_DIR="$SCRIPT_DIR/campaign_ctd2_friction_0_01/vec_$(printf '%03d' "$INDEX")"

if [ -f "$SAMPLE_DIR/rg.txt" ]; then
    mkdir -p "$SAMPLE_DIR"
    echo "vec_${INDEX}: already complete, nothing to do"
    exit 0
fi
mkdir -p "$SAMPLE_DIR"

source "$HOME/miniforge/bin/activate" "$HOME/envs/bff"
cd "$REPO"
export PYTHONPATH="$REPO"
export OPENMM_CPU_THREADS="$SLURM_CPUS_PER_TASK"

echo "vec_${INDEX}: job $SLURM_JOB_ID on $(hostname)"
python - <<'PY' || true
import openmm as mm
for i in range(mm.Platform.getNumPlatforms()):
    print(f"  platform: {mm.Platform.getPlatform(i).getName()}")
PY

python -m examples.calvados_gmx.run_ctd2_friction_campaign "$INDEX"

if [ -f "$SAMPLE_DIR/rg.txt" ]; then
    echo "vec_${INDEX}: complete"
else
    echo "vec_${INDEX}: incomplete"
fi

#!/bin/bash
#SBATCH --job-name=peptide
#SBATCH --constraint=gen-b
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1


function resubmit {
    if [ ! -f "confout.gro" ]; then
       sbatch --export=ALL,job_restarted=1 run.sh
    fi
}

trap resubmit EXIT

if [ -z ${job_restarted+x} ]; then
    source /uochb/soft/b/spack/20221211-git/share/spack/setup-env.sh
    spack load gromacs@2023.1
fi


echo "JOB TITLE STARTED: $(date) AT NODE: $(hostname) JOBID:" ${SLURM_JOB_ID}


# Start gromacs Run
gmx mdrun -cpi -v -maxh 4

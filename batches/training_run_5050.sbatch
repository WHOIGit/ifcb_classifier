#!/usr/bin/env bash

#SBATCH --job-name=June17_5050
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10240
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sbatchelder@whoi.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-18
#SBATCH --output=output/slurm-out/%x_%a.out

cd /vortexfs1/scratch/sbatchelder/ifcb

echo "Job ID: $SLURM_JOB_ID, JobName-ArrayID: $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID"
hostname; pwd; date

module load cuda91/toolkit cuda91/blas cuda91/cudnn cuda91/fft
module load anaconda
source activate ifcb-matrix 

echo "Environment... Loaded"

## SETUP OUTPUT DIRECTORY ##
if ["$SLURM_ARRAY_TASK_ID" -gt 9]
  then
    ARR_ID=$((SLURM_ARRAY_TASK_ID-9))
    OUTDIR=output/2018-06-17_5050_B/June17_B_0"$ARR_ID"
    mkdir -vp "$OUTDIR"
    time python neuston_net.py June17_5050/test June17_5050/train "$OUTDIR" --augment flipxy
  else
    ARR_ID="$SLURM_ARRAY_TASK_ID"
    OUTDIR=output/2018-06-17_5050_A/June17_A_0"$ARR_ID"
    mkdir -vp "$OUTDIR"
    time python neuston_net.py June17_5050/train June17_5050/test "$OUTDIR" --augment flipxy
fi


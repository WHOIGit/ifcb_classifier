#!/usr/bin/env bash

#SBATCH --job-name=June17_8020
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10240
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sbatchelder@whoi.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=output/slurm-out/%x_%a.out

cd /vortexfs1/scratch/sbatchelder/ifcb

echo "Job ID: $SLURM_JOB_ID, JobName-ArrayID: $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID"
hostname; pwd; date

module load cuda91/toolkit cuda91/blas cuda91/cudnn cuda91/fft
module load anaconda
source activate ifcb-matrix 

echo "Environment... Loaded"

## SETUP OUTPUT DIRECTORY ##
OUTDIR=output/2018-06-17_8020/June17_0"$SLURM_ARRAY_TASK_ID"
mkdir -vp "$OUTDIR"

## RUN THE TRAINING MODEL ##
time python neuston_net.py June17_8020/train June17_8020/test "$OUTDIR" --augment flipxy



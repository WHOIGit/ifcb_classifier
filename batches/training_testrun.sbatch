#!/usr/bin/env bash

#SBATCH --job-name=testrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10240
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sbatchelder@whoi.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=output/slurm-out/test_%x.out

cd /vortexfs1/scratch/sbatchelder/ifcb

echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date

module load cuda91/toolkit cuda91/blas cuda91/cudnn cuda91/fft
module load anaconda
source activate ifcb_new

echo "Environment... Loaded"

## SETUP OUTPUT DIRECTORY ##
OUTDIR=output/new-conda-test
mkdir -vp "$OUTDIR"

## RUN THE TRAINING MODEL ##
time python neuston_net.py least24_5050/train least24_5050/test "$OUTDIR" --augment flipxy --model squeezenet


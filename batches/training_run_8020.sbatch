#!/usr/bin/env bash

#SBATCH --job-name=Aug29-8020
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
source activate ifcb 

echo "Environment... Loaded"

## SETUP OUTPUT DIRECTORY ##

OUTDIR=output/Aug29_8020/Aug29_0"$SLURM_ARRAY_TASK_ID"
mkdir -vp "$OUTDIR"
time ./neuston_net.py from-sosiknas/Aug29_fullset $OUTDIR --split 80:20 --class-config master_classlist.csv mvco_northeastshelf_train_cfg --class-minimum 30 --model inception_v3 --pretrained --augment flipxy --seed 8020



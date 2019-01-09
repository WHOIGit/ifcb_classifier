#!/usr/bin/env bash

#SBATCH --job-name=
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10240
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sbatchelder@whoi.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3

date; pwd
echo "Job ID: $SLURM_JOB_ID, Job Name: $SLURM_JOB_NAME"

SAVEDIR=$SLURM_JOB_NAME
cd /vortexfs1/scratch/sbatchelder/ifcb
mkdir -v output/$SAVEDIR
mv slurm-$SLURM_JOB_ID.out output/$SAVEDIR/

date;hostname;pwd

module load anaconda
module load cuda91/toolkit cuda91/blas cuda91/cudnn cuda91/fft

source activate ifcb-matrix  # TODO: check conda/pip versioning

echo "Environment... Loaded"

## RUN THE TRAINING MODEL ##
time python neuston_net.py ifcb_top50_train_80 ifcb_top50_test_20 $SAVEDIR --model squeezenet


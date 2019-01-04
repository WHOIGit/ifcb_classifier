#!/usr/bin/env bash

#SBATCH --job-name=vgg16
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40000
#SBATCH --time=23:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sbatchelder@whoi.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

date; pwd
echo "Job ID: $SLURM_JOB_ID, Job Name: $SLURM_JOB_NAME"

SAVEDIR=$SLURM_JOB_NAME

mkdir -v output/$SAVEDIR
mv slurm-$SLURM_JOB_ID.out output/$SAVEDIR/

date;hostname;pwd

module load anaconda
module load cuda91/toolkit cuda91/blas cuda91/cudnn cuda91/fft

source activate ifcb-matrix

echo "Environment... Loaded"

training_dir="ifcb_top50_train_80"
testing_dir="ifcb_top50_test_20"

## RUN THE TRAINING MODEL ##
time python neuston_net.py $training_dir $testing_dir $SAVEDIR --epochs 60 --model vgg16


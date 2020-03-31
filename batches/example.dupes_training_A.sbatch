#!/usr/bin/env bash

#SBATCH --job-name=TRAIN_ExampleDupes_A
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10240
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=username@whoi.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=slurm-logs/%A_%a.%x.out

# SETTING OPERATIVE DIRECTORY #
cd /vortexfs1/scratch/username/ifcb

# LOGGING JOB DETAILS #
echo "Job ID: $SLURM_JOB_ID, JobName-ArrayID: $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID"
hostname; pwd; date

# SETTING UP ENVIRONMENT #
module load cuda91/toolkit cuda91/blas cuda91/cudnn cuda91/fft
module load anaconda
source activate ifcb
echo "Environment... Loaded"

# PARAMS #
OUTDIR=training-output/ExampleDupes/ExampleDupes_A_0"$SLURM_ARRAY_TASK_ID"
MODEL=inception_v3
DATASET=training-data/ExampleDataset
mkdir -vp "$OUTDIR"

# RUN SCRIPT #
time neuston_net.py "$DATASET" "$OUTDIR" --split 50:50 --seed 5050 --model $MODEL --pretrained --augment flipxy

# to further configur the combining/skipping of certain classes, use
# --class-config path/to/ExampleDataset-classlist-config.csv your-configuration
# --class-minimum N

# Note: --seed must be the same as TRAIN-ExampleDupes-5050B
#       --class-config and --class-minimum must also be the same as for 5050B
# Note: --swap is NOT included. This is the main difference between 5050A and 5050B

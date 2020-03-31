#!/usr/bin/env bash

#SBATCH --job-name=ArrayTRAIN_Example
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10240
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=username@whoi.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-3
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

# SET PARAMS BASED ON ARRAY ID #
case $SLURM_ARRAY_TASK_ID in

  1)
    MODEL=alexnet
    ;;

  2)
    MODEL=squeezenet
    ;;

  3)
    MODEL=vgg11_bn
    ;;

  *)
    echo "BAD CASE-STATE"
    ;;
esac

# STATIC PARAMS #
DATASET=training-data/ExampleTrainingData
OUTDIR=training-output/TrainedExample_Array/Example_0"$SLURM_ARRAY_TASK_ID"
mkdir -vp "$OUTDIR"

# RUN SCRIPT #
time python neuston_net.py "$DATASET" "$OUTDIR" --split 80:20 --seed 8020 --model $MODEL --pretrained --augment flipxy

# This keeps a number of default parameters at their default values, such as
# --min-epochs 16
# --max-epochs 60
# --batch-size 108
# --loaders 4

# to further configure the combining/skipping of certain classes, use
# --class-config path/to/ExampleDataset-classlist-config.csv your-configuration
# --class-minimum N


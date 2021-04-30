#!/usr/bin/env bash
#SBATCH --job-name=RUN_ExampleDataset_ExampleModel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10240
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=username@whoi.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-logs/%j.%x.out

# SETTING OPERATIVE DIRECTORY #
cd /vortexfs1/scratch/username/ifcbnn

# LOGGING JOB DETAILS #
echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date

# SETTING UP ENVIRONMENT #
module load cuda10.1/toolkit cuda10.1/blas cuda10.1/cudnn/8.0.2 cuda91/fft
module load anaconda
source activate ifcbnn
echo "Environment... Loaded"

# PARAMS #
RUN_ID=ExampleRunID
MODEL=training-output/TrainedExample/TrainedExample.ptl
DATASET=run-data/ExampleDataset

# RUN SCRIPT #
python ./neuston_net.py RUN "$DATASET" "$MODEL" "$RUN_ID"

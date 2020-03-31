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

echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date

cd /vortexfs1/scratch/username/ifcb

module load cuda91/toolkit cuda91/blas cuda91/cudnn cuda91/fft
module load anaconda
source activate ifcb

echo "Environment... Loaded"

## PARAMS ##
MODEL=training-output/TrainedExample/model.pt
DATASET=run-data/ExampleDataset
OUTDIR=run-output/"$SLURM_JOB_NAME"
mkdir -vp "$OUTDIR"

time python ./neuston_run.py "$DATASET" --model "$MODEL" --outdir "$OUTDIR"

# This keeps a number of default parameters at their default values, such as
# --input-type bins
# --outfile {bin}_class_v2.h5
# --batch-size 108
# --loaders 4

# if you want to limit your processing to a certain selection of bins, you may use
# --bin-filter path/to/your-list-of-bins.txt
# where "your-list-of-bins.txt" has one bin-id per line. eg: "IFCB5_2017_338_173613"

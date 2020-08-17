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
cd /vortexfs1/scratch/username/ifcb

# LOGGING JOB DETAILS #
echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date

# SETTING UP ENVIRONMENT #
module load cuda10.1/toolkit cuda10.1/blas cuda10.1/cudnn cuda91/fft
module load anaconda
source activate ifcbnn
echo "Environment... Loaded"

# PARAMS #
MODEL=training-output/TrainedExample/model.pt
DATASET=run-data/ExampleDataset
OUTDIR=run-output/"$SLURM_JOB_NAME"
mkdir -vp "$OUTDIR"

# RUN SCRIPT #
time python neuston_net.py RUN "$MODEL" "$DATASET" "$OUTDIR" --type bin --outfile "{bin}_class_v2.h5"



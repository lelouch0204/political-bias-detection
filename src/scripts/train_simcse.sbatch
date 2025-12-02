#!/bin/bash
# SBATCH directives
#SBATCH --job-name=train_simcse
#SBATCH --output=slurm_logs/train_simcse_%j.out
#SBATCH --error=slurm_logs/train_simcse_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nparikh44@gatech.edu

cd /home/hice1/nparikh44/ml-project/cs-7641-group-29/src

# Change these if you use a different conda env or python path
CONDA_ENV_NAME=simcse-h100

# The repository root (script paths are relative to this file location)
# REPO_ROOT=$(realpath "$(dirname "$0")/../..")

# echo "Running on host: $(hostname)"
# echo "Job id: ${SLURM_JOB_ID}"
# echo "Repository root: ${REPO_ROOT}"

# mkdir -p "${REPO_ROOT}/slurm_logs"

# Load modules if your cluster uses module system (uncomment if needed)
module load cuda/12.1
module load anaconda3


# Ensure CUDA envvars recommended by README (adjust if needed)
# export CUDA_HOME=${CONDA_PREFIX:-$CUDA_HOME}
# export CPLUS_INCLUDE_PATH=${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH}
# export LIBRARY_PATH=${CUDA_HOME}/lib:${LIBRARY_PATH}

# cd "${REPO_ROOT}"
echo "Starting training script..."
echo "Job started in directory: $(pwd)"
export CONDA_DIR="$HOME/scratch/miniconda"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate simcse-h100

export PATH="${CONDA_PREFIX}/bin:${PATH}"

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
echo "Python executable: $(which python)"
echo "Python path:"
python -c "import sys; print('\n'.join(sys.path))"

python train/unsup_simcse_train.py --train_file data/political_corpus.txt --model_name roberta-base \
  --batch_size 128 --epochs 3 --output_dir ~/scratch/experiments/simcse_political_unsup_roberta_base/_3_epochs

echo "Job finished"
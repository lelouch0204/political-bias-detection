#!/bin/bash
# SBATCH directives
#SBATCH --job-name=train_simcse
#SBATCH --output=slurm_logs/train_simcse_sup_%j.out
#SBATCH --error=slurm_logs/train_simcse_sup_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H100:3
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

# Multi-GPU NCCL settings for single-node runs
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
NUM_GPUS=${SLURM_GPUS_ON_NODE:-3}

export PATH="${CONDA_PREFIX}/bin:${PATH}"

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
echo "Python executable: $(which python)"
echo "Python path:"
python -c "import sys; print('\n'.join(sys.path))"

# python train/sup_simcse_train.py \
#   --json_path ./data/allsides/ordinal_simcse_pairs.json \
#   --output_dir ./.experiments/ordinal_simcse/all-mpnet-base-v2 \
#   --model_name sentence-transformers/all-mpnet-base-v2 \
#   --batch_size 128 \
#   --accum_steps 8 \
#   --lr 2e-5 \
#   --epochs 5 \
#   --max_len 128 \
#   --device cuda

accelerate launch train/train_ordinal_simcse_accelerate.py \
  --json_path ./data/allsides/ordinal_simcse_pairs.json \
  --output_dir ./.experiments/ordinal_simcse_multi/all-mpnet-base-v2 \
  --model_name sentence-transformers/all-mpnet-base-v2 \
  --batch_size 128 \
  --accum_steps 8 \
  --lr 2e-5 \
  --epochs 20 \
  --max_len 128 

echo "Job finished"
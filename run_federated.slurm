#!/bin/sh
#SBATCH --job-name=run-10-june-without-cross-edges
#SBATCH --account=ewi-st-dis
#SBATCH --qos=short
#SBATCH --partition=ewi-st       # Request partition
#SBATCH --time=4:00:00           # Wall-clock time
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Increase file descriptor limit (based on Cagri's version)
ulimit -n 65536

# Put Apptainer cache on /tmp so HOME quota doesn't explode
export APPTAINER_CACHEDIR=/tmp/$USER

# Paths 
APPTAINER_IMAGE="/tudelft.net/staff-umbrella/CSE3000GLTD/sceydeli/pytorch2.2.2-cuda11.8-ubuntu22.04-fraudGT_v2.sif"
CODE_DIR="/home/nfs/sceydeli/thesis/Subgraph_Federated_Learning"
HOST_BASE_DIR="/tudelft.net/staff-umbrella/CSE3000GLTD/sceydeli"

# Make sure log directory exists (on the host)
mkdir -p "${CODE_DIR}/logs/slurm"

# Optional module load
# module load cuda/12.4

# Run the training inside the container
srun apptainer exec \
  --nv \
  --containall \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${HOST_BASE_DIR}:${HOST_BASE_DIR}" \
  "${APPTAINER_IMAGE}" \
  bash -c "cd /workspace && export PYTHONPATH=/workspace:\$PYTHONPATH && python -m scripts.training.train_local_baseline"

#!/bin/bash

#SBATCH --output=/nese/mit/group/evlab/u/luwo/projects/prosody/log/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/nese/mit/group/evlab/u/luwo/projects/prosody/log/%j.err  # where to store error messages

#SBATCH -t 9:00:00 
#SBATCH -N 1                  # one node
#SBATCH -c 1                   # 4 virtual CPU cores
#SBATCH --gres=gpu:1
#### SBATCH --constraint=any-A100
#SBATCH --mem=20G             # 40 GB of RAM

#### srun -n 4 -t 09:00:00 --mem=30G --gres=gpu:1 --constraint=any-A100 --pty bash
#### srun -n 4 -t 09:00:00 --mem=30G --gres=gpu:QUADRORTX6000:1 --pty bash


# # Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

echo "Running on node: $(hostname)"

# Binary or script to execute
python src/train.py experiment=prominence_regression_absolute_gpt2

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

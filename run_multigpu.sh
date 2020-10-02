#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=TOY-GAN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --account=machnitz
#SBATCH --partition=pGPU
#SBATCH --exclusive

module load compilers/cuda/11.0
export CUDA_VISIBLE_DEVICES="0"
nvidia-smi
srun /gpfs/home/machnitz/miniconda3/envs/pytorch/bin/python main.py --batch_size 128 --learning_rate 0.0001 --data_path '/gpfs/work/machnitz/plankton_dataset/VPR_M87_grouped' --data_is_grouped

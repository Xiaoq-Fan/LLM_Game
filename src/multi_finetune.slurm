#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:3

module purge
module load GCC/12.3.0 Python/3.11.3 OpenMPI/4.1.5 CUDA/11.7.0 
        # Number of inputs
source ~/oldversion/bin/activate
# example_mp.py sets pool size from the first argument
python Run_all.py

#!/bin/bash

#SBATCH -A try25_mscuttari
#SBATCH -p boost_usr_prod
#SBATCH --time=0:30:00        # format: HH:MM:SS
#SBATCH --nodes=1             # -N nodes
#SBATCH --ntasks-per-node=1   # MPI tasks/node (max 32)
#SBATCH --mem=16GB            # memory/node (max 481GB, 4 GB/core)
#SBATCH --gres=gpu:1          # number of GPUs per node (max 4)
#SBATCH --job-name=particles_gpu
#SBATCH --error=err-%J.err    # stderr file
#SBATCH --output=out-%J.out   # stdout file

module purge
module load autoload

cd $SLURM_SUBMIT_DIR
module load intel-oneapi-compilers
rm -f Particles2D_2024

nvcc -O3 Particles2D_2024.cu -o Particles2D_2024 -Xcompiler=-fopenmp
mpirun --mca mpi_yield_when_idle 1 --bind-to none -n 1 ./Particles2D_2024

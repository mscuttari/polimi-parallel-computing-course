#!/bin/bash

#SBATCH -A try25_scuttari
#SBATCH -p boost_usr_prod
#SBATCH --time=0:30:00        # format: HH:MM:SS
#SBATCH --nodes=1             # -N nodes
#SBATCH --ntasks-per-node=1   # MPI tasks/node (max 32)
#SBATCH --mem=16GB            # memory/node (max 481GB, 4 GB/core)
#SBATCH --gres=gpu:1          # number of GPUs per node (max 4)
#SBATCH --job-name=particles_gpu
#SBATCH --error=stderr.log    # stderr file
#SBATCH --output=stdout.log   # stdout file

module purge
module load cmake
module load nvhpc
module load openmpi
module load intel-oneapi-compilers

cd $SLURM_SUBMIT_DIR || exit 1

make clean
make
mpirun --mca mpi_yield_when_idle 1 --bind-to none -n 1 ./Particles2D_2024
make tar

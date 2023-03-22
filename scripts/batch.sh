#!/bin/bash
#SBATCH -A a08trb35
#SBATCH -p m100_usr_prod
#SBATCH --time 00:10:00      # format: HH:MM:SS
#SBATCH -N 1                 # 1 node
#SBATCH --ntasks-per-node=32 # 8 tasks out of 128
#SBATCH --gres=gpu:0         # 1 gpus per node out of 4
#SBATCH --mem=7100           # memory per node out of 246000MB
#SBATCH --job-name=wanda_parallel_multiplication
#SBATCH --mail-type=ALL
#SBATCH --mail-user=walter.nadalin@studenti.units.it
mpirun ./myexecutable       #in case you compiled with spectrum-mpi

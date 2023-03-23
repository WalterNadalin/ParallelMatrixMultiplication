#!/bin/bash
#SBATCH -A tra23_units
#SBATCH -p m100_usr_prod
#SBATCH --time 00:15:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --ntasks-per-node=128 # 8 tasks out of 128
#SBATCH --gres=gpu:4          # 1 gpus per node out of 4
#SBATCH --mem=246000          # memory per node out of 246000MB
#SBATCH --job-name=wanda_parallel_multiplication
#SBATCH --mail-type=ALL
#SBATCH --mail-user=walter.nadalin@studenti.units.it
module load autoload spectrum_mpi
make
dim=3000
prc=2

for value in {1..5}
do
	mpirun -np $prc ./multiplication.x $dim 
	((prc*=2))
done

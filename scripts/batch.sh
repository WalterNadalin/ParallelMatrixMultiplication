#!/bin/bash

#SBATCH -A tra23_units
#SBATCH -p m100_usr_prod
#SBATCH --time 00:30:00       # format: HH:MM:SS
#SBATCH -N 2                  # nodes
#SBATCH --ntasks-per-node=32  # tasks out of 128
#SBATCH --gres=gpu:4          # gpus per node out of 4
#SBATCH --mem=246000          # memory per node out of 246000MB
#SBATCH --ntasks-per-core=1
#SBATCH --job-name=wanda_parallel_multiplication
#SBATCH --mail-type=ALL
#SBATCH --mail-user=walter.nadalin@studenti.units.it

module load autoload spectrum_mpi
module load autoload openblas

for dim in {5000..5000..5000}
do
	make
	prc=32

	for _ in {1..4}
	do
		#mpirun -np $prc -npernode 32 --map-by socket --bind-to core ./multiplication.x $dim 
		((prc*=2))
	done

	make clean
	make flag=dgemm
	export OMP_NUM_THREADS=1
	prc=32

	for value in {1..4}
	do
		#mpirun -np $prc -npernode 32 --map-by socket --bind-to core ./multiplication.x $dim 
		((prc*=2))
	done

	make clean
	make cuda
	prc=4

	for value in {1..2}
	do
		mpirun -np $prc -npernode 4 -npersocket 2 --bind-to core ./multiplication.x $dim 
		((prc*=2))
	done

	make clean
done

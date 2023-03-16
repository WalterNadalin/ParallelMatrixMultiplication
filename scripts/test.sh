#!/bin/bash

module load autoload python/3.7.7
module load autoload spectrum_mpi

make
python test/create.py $1
mpirun -np $2 ./multiplication.x
python test/test.py

#!/bin/bash

module load autoload spectrum_mpi
make
mpirun -np $1 ./multiplication.x

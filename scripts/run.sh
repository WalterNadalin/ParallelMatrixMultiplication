#!/bin/bash

module load autoload spectrum_mpi
make
mpirun -np 4 ./multiplication.x

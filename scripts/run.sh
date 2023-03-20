#!/bin/bash

#module load autoload spectrum_mpi
#module load autoload python/3.7.7

if [ $# -eq 0 ]
then
  echo Arguments missing: dimension '('optional')' and/or number of processors '('mandatory')'
fi

if [ $# -eq 1 ]
then
  make
  mpirun -np $1 ./multiplication.x
fi

if [ $# -gt 1 ]
then
  make
  python ./test/create.py $1
  mpirun -np $2 ./multiplication.x
fi

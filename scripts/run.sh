#!/bin/bash

module load autoload spectrum_mpi

if [ $# -lt 2 ]
then
  echo
  echo Arguments missing:
  echo  + number of processors '('mandatory')'
  echo  + dimension of the square matrices  '('mandatory')'
  echo  + -DDEBUG flag for the compiler '('optional')'
  echo
  echo For example to run the program with n times n matrices on m processors and include
  echo debugging option '('that is, print matrices generated and result on some files')' use the
  echo command:
  echo
  echo  $ bash ./script/run.sh m n -DDEDUG
  echo
fi

if [ $# -eq 2 ]
then
  make
  mpirun -np $1 ./multiplication.x $2
fi

if [ $# -gt 2 ]
then
  make debug=$3
  mpirun -np $2 ./multiplication.x $2
fi

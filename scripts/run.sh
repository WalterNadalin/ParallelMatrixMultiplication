#!/bin/bash
module load autoload spectrum_mpi
module load autoload openblas
module load autoload cuda

if [ $# -lt 2 ]
then
  echo
  echo Arguments missing:
  echo  + number of processors '('mandatory')'
  echo  + dimension of the square matrices  '('mandatory')'
  echo  + '`'debug'`' flag for the compiler '('optional')'
  echo
  echo For example to run the program with 16 times 16 matrices among 4 processes and include
  echo debugging option '('that is, print matrices generated and result on some files')' use
  echo the command:
  echo
  echo  $ bash ./script/run.sh 4 16 debug
  echo
elif [ $# -eq 2 ]
then
  make
  mpirun -np $1 ./multiplication.x $2
elif [ $# -eq 3 ]
then
  export OMP_NUM_THREADS=1

  if [ $3 = debug ]
  then
    make flag=$3
    mpirun -np $1 ./multiplication.x $2
  else
    make $3
    mpirun -np $1 ./multiplication.x $2
  fi
else
  export OMP_NUM_THREADS=1 
  make $3 flag=$4
  mpirun -np $1 ./multiplication.x $2
fi

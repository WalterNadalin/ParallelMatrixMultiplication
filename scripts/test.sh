#!/bin/bash

#module load autoload python/3.7.7
#module load autoload spectrum_mpi

make
python test/create.py $1
mpirun -np $2 ./multiplication.x
gcc test/serial_multiplication.c -o test/serial.x
./test/serial.x
rm ./test/serial.x

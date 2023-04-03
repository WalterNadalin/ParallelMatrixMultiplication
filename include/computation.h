#ifndef COMPUTATION_H_INCLUDE
#define COMPUTATION_H_INCLUDE

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void serial_multiplication(double *, double *, double *, int, int, int); // Matrix multiplication
void parallel_multiplication(double *, double *, double *, int, float *, float *);
void gather_multiplication(double *, double *, double *, int, int, int, float *, float *, double *, int *, int *, MPI_Datatype);

#endif

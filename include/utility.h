#ifndef UTILITY_H_INCLUDE
#define UTILITY_H_INCLUDE

#include<mpi.h>
#include<stdio.h>

void print(double*, int, int); // Prints a matrix
void fill(double*, int, int, double); // Fill a matrix with a given value
void distributed_print(double*, int, int, int, int); // Prints a distributed matrix

#endif

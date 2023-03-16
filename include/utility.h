#ifndef UTILITY_H_INCLUDE
#define UTILITY_H_INCLUDE

#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>

void print(double*, int, int, char *); // Prints a matrix
void fill(double*, int, int, double); // Fill a matrix with a given value
void distributed_print(double*, int, int, int, int, char*); // Prints a distributed matrix
void get_dimension(int, int, int *, char *); // Reads and share the dimension
void get_slices(double *, double *, double *, double *, int, int, int, int, char *);

#endif

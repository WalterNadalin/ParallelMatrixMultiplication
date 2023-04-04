#ifndef PARALLELIO_H_INCLUDE
#define PARALLELIO_H_INCLUDE

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void distributed_print(double *, int, int, int, char *); // Prints a distributed matrix
void get_dimension(int, int *, char *); // Reads and share the dimension of the matrices
void get_slices(double *, double *, int, char *); // Generates horizontal slices
void generate_slices(double *, double *, int, int); // Reads and scatters horizontal slices

#endif

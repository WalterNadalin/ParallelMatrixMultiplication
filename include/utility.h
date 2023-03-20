#ifndef UTILITY_H_INCLUDE
#define UTILITY_H_INCLUDE

#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

void print(double *, int, int, FILE *); // Prints a matrix
void distributed_print(double *, int, int, char *); // Prints a distributed matrix
void get_dimension(int, int *, char *); // Reads and share the dimension
void get_slices(double *, double *, int, int, int, char *);
void get_counts(int *, int *, int, int);

#endif

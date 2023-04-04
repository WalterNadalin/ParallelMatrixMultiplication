#ifndef UTILITY_H_INCLUDE
#define UTILITY_H_INCLUDE

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define RED "\x1b[31m"
#define GREEN "\x1b[32m"
#define NORMAL "\x1b[m"

double randfrom(double, double); // Generates a random double
void print(double *, int, int, FILE *); // Prints a matrix
void test(char *, char *); // Check the correctness of the result
void generate_matrices(int, char *); // Generates a writes on file two matrices

#endif

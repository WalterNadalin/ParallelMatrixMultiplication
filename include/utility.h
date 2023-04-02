#ifndef UTILITY_H_INCLUDE
#define UTILITY_H_INCLUDE

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define RED "\x1b[31m"
#define GREEN "\x1b[32m"
#define NORMAL "\x1b[m"

void print(double *, int, int, FILE *); // Prints a matrix
double randfrom(double, double); // Generates a random double
void test(char *, char *); // Check the correctness of the result

#endif

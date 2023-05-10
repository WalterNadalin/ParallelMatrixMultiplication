#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "utility.h"
#include "parallelio.h"
#include "computation.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int n, id, prc, loc, rst; // Rank, number of processors and other things
  float first, second, io_time = 0, cp_time = 0; // For time measures
  double *A, *B, *C; // Matrices to multiply, result and current vertical slice
  char *p, *data = "data/matrices.txt", *times = "data/times.txt";
  FILE *file;

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);

  if(!strcmp(argv[1], "read")) { // Reading and parsing command line arguments
    get_dimension(0, &n, data);    

    if(argc > 2 && id == 0) 
      printf("\n\tWarning: option `read` chosen, the dimension passed will be ignored\n");
  } else if(!strcmp(argv[1], "generate")) {
    n = strtol(argv[2], &p, 10);
  } else if(id == 0) {
    printf("\n\tError: unknown option `%s` passed, aborting.\n\n", argv[1]);
    return 1;
  }

  if(n < prc && id == 0) {
    printf("\n\tError: number of processes (%d) greater than dimension (%d), aborting.\n\n", 
           prc, n);
    return 2;
  }

  rst = n % prc;
  loc = (id < rst) ? n / prc + 1 : n / prc; // Local rows number of the horizontal slices

  A = (double *) malloc(n * loc * sizeof(double));
  B = (double *) malloc(n * loc * sizeof(double));
  C = calloc(n * loc, sizeof(double));

  first = MPI_Wtime();
  
  if(!strcmp(argv[1], "read")) {
    get_slices(A, B, 0, data); // Get matrices written on the file
  } else {
    generate_slices(A, B, n, loc); // Creates scattered slices of matrices A and B
  }

  second = MPI_Wtime();

  parallel_multiplication(A, B, C, n, &io_time, &cp_time); // Multiplication

#ifdef DEBUG // Print matrices A, B and C
  char *result = "data/result.txt";

  if(id == 0) {
    file = fopen(data, "w");
    fprintf(file, "%d\n", n);
    fclose(file);
  }
  
  distributed_print(A, n, loc, 0, data);
  distributed_print(B, n, loc, 0, data);
  distributed_print(C, n, loc, 1, result);

  if(id == 0) test(data, result); // Check the correctness of the result
#endif // 'gg', cit. Gallo

  if(id == 0) { // Print execution times
#ifdef DGEMM 
    char *sign = "dgemm";
#elif CUDA
    char *sign = "cudgemm";
#else
    char *sign = "naive";
#endif

    printf("\n\tVersion: %s\n\tDimension of the matrices: %d\n\tNumber of processes: %d\
	    \n\n\tGeneration/reading time: %lf\n\tCommunication time: %lf\
	    \n\tComputation time: %lf\n\n", sign, n, prc, second - first, io_time, cp_time);
    
    file = fopen(times, "a");
    fprintf(file, "%s %d %d %lf %lf\n", sign, n, prc, io_time, cp_time);
    fclose(file);
  }
  
  free(A);
  free(B);
  free(C);
  MPI_Finalize();

  return 0;
}

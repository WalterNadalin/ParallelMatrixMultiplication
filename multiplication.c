#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "utility.h"
#include "parallelio.h"
#include "computation.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int n, id, prc, loc, rst; // Rank, number of processors, ...
  float first, second, io_time = 0, cp_time = 0; // For time measures
  double *A, *B, *C; // Matrices to multiply, result and current vertical slice
  char *p, *data = "data/matrices.txt", *times = "data/times.txt";
  FILE *file;

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);

  n = strtol(argv[1], &p, 10);

  MPI_Barrier(MPI_COMM_WORLD);
  first = MPI_Wtime();
  if(id == 0) generate_matrices(n, data);
  MPI_Barrier(MPI_COMM_WORLD);
  second = MPI_Wtime();
  
  if(id == 0) printf("%f \n", second - first);
  
  rst = n % prc;
  loc = (id < rst) ? n / prc + 1 : n / prc; // Local rows number of the horizontal slices

  MPI_Barrier(MPI_COMM_WORLD);
  first = MPI_Wtime();

  A = (double *) malloc(n * loc * sizeof(double));
  B = (double *) malloc(n * loc * sizeof(double));
  C = calloc(n * loc, sizeof(double));

  //generate_slices(A, B, n, loc); // Creates scattered slices of matrices A and B
  get_slices(A, B, 0, data);
  
  MPI_Barrier(MPI_COMM_WORLD);
  second = MPI_Wtime();
  
  io_time += second - first;

  parallel_multiplication(A, B, C, n, &io_time, &cp_time); // Naive or dgemm parallel multiplication

//  if(id == 0) printf("%f \n", second - first);
#ifdef DEBUG // Print matrices A, B and C
  char *result = "data/result.txt";
/*
  if(id == 0) {
    file = fopen(data, "w");
    fprintf(file, "%d\n", n);
    fclose(file);
  }
  
  distributed_print(A, n, loc, 0, data);
  distributed_print(B, n, loc, 0, data);*/
  distributed_print(C, n, loc, 1, result);

  if(id == 0) test(data, result);
#endif // 'gg', cit. Gallo

  if(id == 0) {
#ifdef DGEMM 
    char *sign = "dgemm";
#elif CUDA
    char *sign = "cudgemm";
#else
    char *sign = "naive";
#endif
    
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

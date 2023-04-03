#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "utility.h"
#include "parallelio.h"
#include "computation.h"

#ifdef DGEMM
  #include <cblas.h>
#endif

#ifdef CUDA
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include "cublas_v2.h"
  #include "gpu.h"
#endif

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int n, id, prc, loc, rst, cnt, upr, root = 0; // Rank, number of processors, ...
  int *counts, *displs; // Counts and displacements for `gathering` the buffer
  float first, second, third, io_time = 0, cp_time = 0; // For time measures
  double *A, *B, *C, *bfr; // Matrices to multiply, result and current vertical slice
  char *p, *times = "data/times.txt";
  FILE *file;
  MPI_Datatype cntgs; // Custom datatype to gather the vertical slices of B

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);

  n = strtol(argv[1], &p, 10);

#ifdef DEBUG // Print matrices A and B
  char *data = "data/matrices.txt", *result = "data/result.txt";

  if(id == root) {
    file = fopen(data, "w");
    fprintf(file, "%d\n", n);
    fclose(file);
  }
#endif

  rst = n % prc;
  loc = (id < rst) ? n / prc + 1 : n / prc; // Local rows number of the horizontal slices
  cnt = (rst) ? n / prc + 1 : n / prc; // Global columns number of the vertical slices
  upr = (rst) ? n / cnt : prc; // Total number of vertical slices of B minus one
  
  MPI_Type_vector(loc, cnt, n, MPI_DOUBLE, &cntgs); // To create the vertical slices of B
  MPI_Type_commit(&cntgs);
  
  counts = (int *) malloc(prc * sizeof(int));
  displs = (int *) malloc(prc * sizeof(int));
  A = (double *) malloc(n * loc * sizeof(double));
  B = (double *) malloc(n * loc * sizeof(double));
  C = calloc(n * loc, sizeof(double));
  bfr = (double *) malloc(n * cnt * sizeof(double));

  get_counts(counts, displs, n, cnt);
  generate_slices(A, B, n, loc); // Creates scattered slices of matrices A and B

#ifdef DEBUG
  distributed_print(A, n, loc, 0, data);
  distributed_print(B, n, loc, 0, data);
#endif // 'gg', cit. Gallo

  for(int m = 0; m < upr; m++) {
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    first = MPI_Wtime();
    
    MPI_Allgatherv(B + m * cnt, 1, cntgs, bfr, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    second = MPI_Wtime();
 
#ifdef DGEMM  
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, loc, cnt, n, 1, A, n, bfr, cnt, 0, C + m * cnt, n);
#elif CUDA
    cuda_multiplication(A, loc, n, bfr, cnt, C + m * cnt, &io_time, &cp_time);
#else
    serial_multiplication(A, bfr, C + m * cnt, loc, cnt, n);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    third = MPI_Wtime();
    io_time += second - first; // Communication time need to set and gather the slicie

#ifndef CUDA
    cp_time += third - second; // Computation time of the matrix multiplicatio
#endif
  }

  rst = n % cnt; // Columns of the remaining vertical slice

  // Doing the same thing for the last vertical slice
  if(rst) {
    //rst = n % cnt;
    MPI_Type_free(&cntgs);
    MPI_Type_vector(loc, rst, n, MPI_DOUBLE, &cntgs);
    MPI_Type_commit(&cntgs);

    MPI_Barrier(MPI_COMM_WORLD);
    first = MPI_Wtime();
    
    get_counts(counts, displs, n, rst);
    MPI_Allgatherv(B + upr * cnt, 1, cntgs, bfr, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    second = MPI_Wtime();
    
#ifdef DGEMM  
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, loc, rst, n, 1, A, n, bfr, rst, 0, C + upr * cnt, n);
#elif CUDA
    cuda_multiplication(A, loc, n, bfr, rst, C + upr * cnt, &io_time, &cp_time);
#else
    serial_multiplication(A, bfr, C + upr * cnt, loc, rst, n);
#endif
    
    MPI_Barrier(MPI_COMM_WORLD);
    third = MPI_Wtime();
    io_time += second - first;

#ifndef CUDA
    cp_time += third - second;
#endif
  }
  
#ifdef DEBUG
  distributed_print(C, n, loc, 1, result);
  if(id == root) test(data, result);
#endif

  if(id == root) {
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
  free(bfr);
  free(counts);
  free(displs);
  MPI_Type_free(&cntgs);
  MPI_Finalize();

  return 0;
}

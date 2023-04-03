#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <mpi.h>

void get_counts(int *counts, int *displs, int n, int m) {
  /*
   * Gets the counts of the elements to send to or receive from each process and the displacement at
   * which get from or place to the elements a buffer.
   * */
  int loc, id, prc, rst;

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);

  loc = n / prc;
  rst = n % prc;
  displs[0] = 0;

  for(int i = 0; i < prc - 1; i++) {
    counts[i] = (i < rst) ? m * (loc + 1) : m * loc;
    displs[i + 1] = counts[i] + displs[i];
  }

  counts[prc - 1] = m * loc;
}

extern "C" void cuda_multiplication(double *A, double *B, double *C, int n, float *io_time, float *cp_time) {
  int id, prc, loc, rst, cnt, upr; // Rank, number of processors, ...
  int *counts, *displs; // Counts and displacements for `gathering` the buffer
  double *bfr; // Matrices to multiply, result and current vertical slice
  MPI_Datatype cntgs; // Custom datatype to gather the vertical slices of B
  double *devA, *devB, *devC; // Matrices on the device
  const double alpha = 1, beta = 0; // Parameters for `cublasDgemm`
  cublasHandle_t handle;
  cudaEvent_t start, stop;
  float first, second, third, fourth;

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);

  rst = n % prc;
  loc = (id < rst) ? n / prc + 1 : n / prc; // Local rows number of the horizontal slices
  cnt = (rst) ? n / prc + 1 : n / prc; // Global columns number of the vertical slices
  upr = (rst) ? n / cnt : prc; // Total number of vertical slices of B minus one
  rst = n % cnt; // Columns of the remaining vertical slice
  
  MPI_Type_vector(loc, cnt, n, MPI_DOUBLE, &cntgs); // To create the vertical slices of B
  MPI_Type_commit(&cntgs);
 
  // Allocation
  counts = (int *) malloc(prc * sizeof(int));
  displs = (int *) malloc(prc * sizeof(int));
  bfr = (double *) malloc(n * cnt * sizeof(double));
  cudaMalloc((void**)&devA, n * loc * sizeof(double));
  cudaMalloc((void**)&devB, n * cnt * sizeof(double));
  cudaMalloc((void**)&devC, loc * cnt * sizeof(double));
  
  // Initialization
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cublasCreate(&handle);
  cublasSetMatrix(loc, n, sizeof(double), A, loc, devA, loc);
  get_counts(counts, displs, n, cnt);

  // Computation
  for(int m = 0; m < upr; m++) { 
    MPI_Barrier(MPI_COMM_WORLD);
    first = MPI_Wtime();
    
    MPI_Allgatherv(B + m * cnt, 1, cntgs, bfr, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    cublasSetMatrix(n, cnt, sizeof(double), bfr, n, devB, n); // Load CPU data into GPU 
    
    MPI_Barrier(MPI_COMM_WORLD);
    second = MPI_Wtime();

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cnt, loc, n, &alpha, devB, cnt, devA, n, &beta, devC, cnt); // Compute multiplication 
    
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    third = MPI_Wtime();
    
    cublasGetMatrix(cnt, loc, sizeof(double), devC, cnt, C + m * cnt, n); // Load GPU buffer into CPU 
    
    MPI_Barrier(MPI_COMM_WORLD);
    fourth = MPI_Wtime();
 
    *io_time += (second - first) + (fourth - third);
    *cp_time += third - second;
  }

  // Doing the same thing for the last vertical slice
  if(rst) {
    MPI_Type_free(&cntgs);
    MPI_Type_vector(loc, rst, n, MPI_DOUBLE, &cntgs);
    MPI_Type_commit(&cntgs);
    get_counts(counts, displs, n, rst);
    
    MPI_Barrier(MPI_COMM_WORLD);
    first = MPI_Wtime();
    
    MPI_Allgatherv(B + upr * cnt, 1, cntgs, bfr, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD); 
    cublasSetMatrix(n, rst, sizeof(double), bfr, n, devB, n); // Load CPU data into GPU
    
    MPI_Barrier(MPI_COMM_WORLD);
    second = MPI_Wtime();

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rst, loc, n, &alpha, devB, rst, devA, n, &beta, devC, rst); // Compute multiplication 
    
    MPI_Barrier(MPI_COMM_WORLD);
    third = MPI_Wtime();
    
    cublasGetMatrix(rst, loc, sizeof(double), devC, rst, C + upr * cnt, n); // Load GPU buffer into CPU 
    MPI_Barrier(MPI_COMM_WORLD);
    fourth = MPI_Wtime();
 
    *io_time += (second - first) + (fourth - third);
    *cp_time += third - second;
  }
  
  free(bfr);
  free(counts);
  free(displs);
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
  cublasDestroy(handle);
  MPI_Type_free(&cntgs);
}

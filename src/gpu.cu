#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
//#include "gpu.h"

/*void single_multiplication(double *devA, double *devB, double *devC, int n, float *io_time, float *cp_time)
{
  // Load CPU data into GPU buffers
  double *devA, *devB, *devC; // Matrices on the device
  float time;
  const double alpha = 1, beta = 0; // Parameters for `cublasDgemm`
  cublasHandle_t handle;
  cudaEvent_t start, stop;

  //cudaMalloc((void**)&devA, n * m * sizeof(double));
  cudaMalloc((void**)&devB, n * k * sizeof(double));
  cudaMalloc((void**)&devC, m * k * sizeof(double));
  cublasCreate(&handle);

  cublasSetMatrix(m, n, sizeof(double), A, m, devA, m);
  cublasSetMatrix(n, k, sizeof(double), B, n, devB, n);
	
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, devB, k, devA, n, &beta, devC, k);

  // Load GPU buffer into CPU  
  cublasGetMatrix(k, m, sizeof(double), devC, k, C, n);

  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
  cublasDestroy(handle);
}*/

 /* cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0); 
    cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop); 
    *io_time += time / 1000;*/
    
extern "C" void cuda_multiplication(double *A, double *B, double *C, int n, float *io_time, float *cp_time) {
  int id, prc, loc, rst, cnt, upr; // Rank, number of processors, ...
  int *counts, *displs; // Counts and displacements for `gathering` the buffer
  double *bfr; // Matrices to multiply, result and current vertical slice
  MPI_Datatype cntgs; // Custom datatype to gather the vertical slices of B
  double *devA, *devB, *devC; // Matrices on the device
  const double alpha = 1, beta = 0; // Parameters for `cublasDgemm`
  cublasHandle_t handle;

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);

  rst = n % prc;
  loc = (id < rst) ? n / prc + 1 : n / prc; // Local rows number of the horizontal slices
  cnt = (rst) ? n / prc + 1 : n / prc; // Global columns number of the vertical slices
  upr = (rst) ? n / cnt : prc; // Total number of vertical slices of B minus one
  rst = n % cnt; // Columns of the remaining vertical slice
  
  MPI_Type_vector(loc, cnt, n, MPI_DOUBLE, &cntgs); // To create the vertical slices of B
  MPI_Type_commit(&cntgs);
  
  counts = (int *) malloc(prc * sizeof(int));
  displs = (int *) malloc(prc * sizeof(int));
  bfr = (double *) malloc(n * cnt * sizeof(double));
  cublasSetMatrix(loc, n, sizeof(double), A, loc, devA, loc);

  get_counts(counts, displs, n, cnt);

  for(int m = 0; m < upr; m++) { 
  	MPI_Allgatherv(B + m * cnt, 1, cntgs, bfr, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    cublasSetMatrix(n, cnt, sizeof(double), bfr, n, devB, n); // Load CPU data into GPU buffer
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cnt, loc, n, &alpha, devB, cnt, devA, n, &beta, devC, cnt); // Compute multiplication 
    cublasGetMatrix(cnt, loc, sizeof(double), devC, cnt, C + m * cnt, n); // Load GPU buffer into CPU 
  }

  // Doing the same thing for the last vertical slice
  if(rst) {
    MPI_Type_free(&cntgs);
    MPI_Type_vector(loc, rst, n, MPI_DOUBLE, &cntgs);
    MPI_Type_commit(&cntgs);
    get_counts(counts, displs, n, rst);
    MPI_Allgatherv(B + upr * cnt, 1, cntgs, bfr, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
		cublasSetMatrix(n, rst, sizeof(double), bfr, n, devB, n); // Load CPU data into GPU buffer
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rst, loc, n, &alpha, devB, rst, devA, n, &beta, devC, cnt); // Compute multiplication 
    cublasGetMatrix(rst, loc, sizeof(double), devC, rst, C + m * cnt, n); // Load GPU buffer into CPU 
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

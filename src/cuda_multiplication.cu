#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

extern "C" void cuda_multiplication(double *A, int m, int n, double *B, int k, double *C, float *io_time, float *cp_time)
{
  /* ... load CPU data into GPU buffers a_gpu and b_gpu */
  double *devA, *devB, *devC; // Matrices on the device
  float time;
  const double alpha = 1, beta = 0; // Parameters for `cublasDgemm`
  cublasHandle_t handle;
  cudaEvent_t start, stop;

  cudaMalloc((void**)&devA, n * m * sizeof(double));
  cudaMalloc((void**)&devB, n * k * sizeof(double));
  cudaMalloc((void**)&devC, m * k * sizeof(double));

  cublasCreate(&handle);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0); 
  cublasSetMatrix(m, n, sizeof(double), A, m, devA, m);
  cublasSetMatrix(n, k, sizeof(double), B, n, devB, n);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  *io_time += time / 1000;
	
  cudaEventRecord(start, 0);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, devB, k, devA, n, &beta, devC, k);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  *cp_time += time / 1000;
  
  cudaEventRecord(start, 0);
  cublasGetMatrix(k, m, sizeof(double), devC, k, C, n);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  *io_time += time / 1000;

  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
  cublasDestroy(handle);
}

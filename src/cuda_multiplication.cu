#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

extern "C" void cuda_multiplication(double *A, int m, int n, double *B, int k, double *C)
{
  /* ... load CPU data into GPU buffers a_gpu and b_gpu */
  double *devA, *devB, *devC; // Matrices on the device
  cublasHandle_t handle;
  //cublasStatus_t stat;
  const double alpha = 1, beta = 0; // Parameters for `cublasDgemm`

  cudaMalloc((void**)&devA, n * m * sizeof(double));
  cudaMalloc((void**)&devB, n * k * sizeof(double));
  cudaMalloc((void**)&devC, m * k * sizeof(double));

  cublasCreate(&handle);

  cublasSetMatrix(m, n, sizeof(double), A, m, devA, m);
  cublasSetMatrix(n, k, sizeof(double), B, n, devB, n);

  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, devB, k, devA, n, &beta, devC, k);

 // safecall(cudaThreadSynchronize());
 // safecall(cudaGetLastError());
    
  cublasGetMatrix(k, m, sizeof(double), devC, k, C, n);

  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
  cublasDestroy(handle);
}

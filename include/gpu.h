#ifndef GPU_H_INCLUDE
#define GPU_H_INCLUDE

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

void cuda_multiplication(double *, int, int, double *, int, double *, float *, float *);

#endif

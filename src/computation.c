#include <stdio.h>
#include <stdlib.h>

void serial_multiplication(double *A, double *B, double *C, int dim_a, int dim_b, int dim) {
  int div = dim_b / 4, upr = div * 4, i, j, k;
  double tmp;
  double *bfr = calloc(dim_b, sizeof(double));

  for(i = 0; i < dim_a; i++) {
    for(k = 0; k < dim; k++) {
      tmp = A[k + i * dim];

      for(j = 0; j < upr; j += 4) {
        bfr[j] += tmp * B[k * dim_b + j]; 
        bfr[j + 1] += tmp * B[k * dim_b + j + 1]; 
        bfr[j + 2] += tmp * B[k * dim_b + j + 2]; 
        bfr[j + 3] += tmp * B[k * dim_b + j + 3]; 
      }
      
      for(j = upr; j < dim_b; j++) bfr[j] += tmp * B[k * dim_b + j]; 
    }
    
    for(j = 0; j < upr; j += 4) {
      C[j + i * dim] = bfr[j]; 
      C[j + 1 + i * dim] = bfr[j + 1]; 
      C[j + 2 + i * dim] = bfr[j + 2]; 
      C[j + 3 + i * dim] = bfr[j + 3]; 
      bfr[j] = 0;
      bfr[j + 1] = 0;
      bfr[j + 2] = 0;
      bfr[j + 3] = 0;
    }
      
    for(j = upr; j < dim_b; j++) {
      C[j + i * dim] = bfr[j];
      bfr[j] = 0;
    } 
  }
}

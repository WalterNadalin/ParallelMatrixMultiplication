#include "computation.h"
#include "parallelio.h"

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

void gather_multiplication(double *A, double *B, double *C, int n, int loc, int cnt, float *io_time, float *cp_time, double *bfr, int *counts, int *displs, MPI_Datatype cntgs) {
    float first, second, third; // For time measures

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    first = MPI_Wtime();
    
    MPI_Allgatherv(B, 1, cntgs, bfr, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    second = MPI_Wtime();
 
#ifdef DGEMM  
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, loc, cnt, n, 1, A, n, bfr, cnt, 0, C, n);
#else
    serial_multiplication(A, bfr, C, loc, cnt, n);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    third = MPI_Wtime();
    *io_time += second - first; // Communication time need to set and gather the slice
    *cp_time += third - second; // Computation time of the matrix multiplicatio
}

void parallel_multiplication(double *A, double *B, double *C, int n, float *io_time, float *cp_time) {
  int id, prc, loc, rst, cnt, upr; // Rank, number of processors, ...
  int *counts, *displs; // Counts and displacements for `gathering` the buffer
  double *bfr; // Matrices to multiply, result and current vertical slice
  MPI_Datatype cntgs; // Custom datatype to gather the vertical slices of B
  

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

  get_counts(counts, displs, n, cnt);

  for(int m = 0; m < upr; m++) gather_multiplication(A, B + m * cnt, C + m * cnt, n, loc, cnt, io_time, cp_time, bfr, counts, displs, cntgs);

  // Doing the same thing for the last vertical slice
  if(rst) {
    MPI_Type_free(&cntgs);
    MPI_Type_vector(loc, rst, n, MPI_DOUBLE, &cntgs);
    MPI_Type_commit(&cntgs);
    get_counts(counts, displs, n, rst);
		gather_multiplication(A, B + upr * cnt, C + upr * cnt, n, loc, rst, io_time, cp_time, bfr, counts, displs, cntgs);
  }
  
  free(bfr);
  free(counts);
  free(displs);
  MPI_Type_free(&cntgs);
}

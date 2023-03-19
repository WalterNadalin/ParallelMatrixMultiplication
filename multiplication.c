#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "utility.h"

int main(int argc, char** argv) {
  int n, id, prc, loc, rst, cnt, upr, root = 0; 
  int *counts, *displs;  
  double *A, *B, *C, *bfr;
  char *data = "data/matrices.txt", *result = "data/result.txt";
  MPI_Datatype cntgs;  

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);

  get_dimension(root, &n, data); 
  rst = n % prc;
  loc = (id < rst) ? n / prc + 1 : n / prc;
  cnt = (rst) ? n / prc + 1 : n / prc;
  upr = (rst) ? prc - 1 : prc;
  
  MPI_Type_vector(loc, cnt, n, MPI_DOUBLE, &cntgs);
  MPI_Type_commit(&cntgs);
  
  counts = (int *) malloc(prc * sizeof(int));
  displs = (int *) malloc(prc * sizeof(int));
  A = (double *) malloc(n * loc * sizeof(double));
  B = (double *) malloc(n * loc * sizeof(double));
  C = calloc(n * loc, sizeof(double));
  bfr = (double *) malloc(n * cnt * sizeof(double));

  get_counts(counts, displs, n, cnt);
  get_slices(A, B, root, n, loc, data); 

  for(int m = 0; m < upr; m++) {
    MPI_Allgatherv(B + m * cnt, 1, cntgs, bfr, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    for(int i = 0; i < loc; i++)
      for(int j = 0; j < cnt; j++)
        for(int k = 0; k < n; k++)
          C[m * cnt + j + i * n] += A[k + i * n] * bfr[k * cnt + j]; 
  }
  
  if(rst) {
    rst = n % cnt;
    
    MPI_Type_vector(loc, rst, n, MPI_DOUBLE, &cntgs);
    MPI_Type_commit(&cntgs);
    
    get_counts(counts, displs, n, rst);
    MPI_Allgatherv(B + upr * cnt, 1, cntgs, bfr, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    for(int i = 0; i < loc; i++)
      for(int j = 0; j < rst; j++)
        for(int k = 0; k < n; k++)
          C[upr * cnt + j + i * n] += A[k + i * n] * bfr[k * rst + j]; 
  }
  
  distributed_print(C, n, loc, result);

  free(A);
  free(B);
  free(C);
  free(bfr);
  free(counts);
  free(displs);

  MPI_Finalize();

  return 0;
}

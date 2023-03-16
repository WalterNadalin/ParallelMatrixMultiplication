#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include "utility.h"

int main(int argc, char** argv) {
  int root = 0;
  int n, id, prc, loc;   
  double *a, *b, *A, *B, *C, *bfr;
  MPI_Datatype contiguous;  
  char *name = "data/matrices.txt";

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);

  get_dimension(id, root, &n, name); 
  loc = n / prc;

  A = (double *) malloc(n * loc * sizeof(double));
  B = (double *) malloc(n * loc * sizeof(double));
  C = (double *) malloc(n * loc * sizeof(double));
  bfr = (double *) malloc(n * loc * sizeof(double));
 
  get_slices(a, b, A, B, id, root, n, loc, name); 
  fill(C, n, loc, (double)0);

  MPI_Type_vector(loc, loc, n, MPI_DOUBLE, &contiguous);
  MPI_Type_commit(&contiguous);

  for(int clm = 0; clm < prc; clm++) {
    MPI_Allgather(B + clm * loc, 1, contiguous, bfr, loc * loc, MPI_DOUBLE, MPI_COMM_WORLD);

    for(int i = 0; i < loc; i++)
      for(int j = 0; j < loc; j++)
        for(int k = 0; k < n; k++)
          C[clm * loc + j + i * n] += A[k + i * n] * bfr[k * loc + j]; 
  }

  distributed_print(C, n, loc, prc, id);

  free(A);
  free(B);
  free(C);
  free(bfr);

  MPI_Finalize();

  return 0;
}

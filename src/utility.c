#include "utility.h"

void print(double* A, int n, int m) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) fprintf(stdout, "%.3g ", A[j + (i * n)]);

    fprintf(stdout, "\n");
  }
}

void fill(double* A, int n, int m, double value) {
  for(int i = 0; i < n * m; i++)
    A[i] = value;
}

void distributed_print(double* A, int n, int loc, int prc, int id) {
  if(id == 0) {
    print(A, n, loc);

    for(int i = 1; i < prc; i++) {
      MPI_Recv(A, loc * n, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      print(A, n, loc);
    }
  }
  else MPI_Send(A, loc * n, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
}

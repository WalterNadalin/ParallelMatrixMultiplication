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

void get_dimension(int id, int root, int *n, char *name) {
  if(id == root) {
    FILE *file = fopen(name, "r");
    fscanf(file, "%d", n);
    fclose(file);
  }

  MPI_Bcast(n, 1, MPI_INT, root, MPI_COMM_WORLD);
}

void get_slices(double *a, double *b, double *A, double *B, int id, int root, int n, int loc, char *name) {
  if(id == root) {
    FILE *file = fopen(name, "r");
    a = (double *) malloc(n * n * sizeof(double));
    b = (double *) malloc(n * n * sizeof(double));

    fscanf(file, "%d", &n);

    for (int i = 0; i < n * n; i++) {
      fscanf(file, "%lf", &a[i]);
      printf("%f\n", a[i]);
    }

    for (int i = 0; i < n * n; i++) {
      fscanf(file, "%lf", &b[i]);
    }

    fclose(file);
  }

  MPI_Scatter(a, n * loc, MPI_DOUBLE, A, n * loc, MPI_DOUBLE, root, MPI_COMM_WORLD);
  MPI_Scatter(b, n * loc, MPI_DOUBLE, B, n * loc, MPI_DOUBLE, root, MPI_COMM_WORLD);

  if(id == root) {
    free(a);
    free(b);
  }
}

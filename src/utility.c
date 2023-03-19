#include "utility.h"

void print(double* A, int n, int m, FILE *file) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) fprintf(file, "%lf ", A[i * n + j]);

    fprintf(file, "\n");
  }
}

void fill(double* A, int n, int m, double value) {
  for(int i = 0; i < n * m; i++)
    A[i] = value;
}

void distributed_print(double* A, int n, int m, char *name) {
  int id, prc, loc, rst, cnt; 

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);
	
  loc = n / prc;
  rst = n % prc;

  if(id == 0) {
    FILE *file = stdout;
    double *bfr = (double *)malloc(m * n * sizeof(double));
    int flag = strcmp(name, "stdout");

    if(flag) {
      fclose(fopen(name, "w"));
      file = fopen(name, "a");
    }

    print(A, n, m, file);

    for(int i = 1; i < prc; i++) {
      cnt = (i < rst) ? (loc + 1) : loc;
      MPI_Recv(bfr, m * n, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      print(bfr, n, cnt, file);
    }

    if(flag) fclose(file);
  }
  else MPI_Send(A, m * n, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
}

void get_dimension(int root, int *n, char *name) {
  int id; 

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  
  if(id == root) {
    FILE *file = fopen(name, "r");
    fscanf(file, "%d", n);
    fclose(file);
  }

  MPI_Bcast(n, 1, MPI_INT, root, MPI_COMM_WORLD);
}

void get_counts(int *counts, int *displs, int n, int m) {
  int loc, id, prc, rst;

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);
  
  loc = n / prc;
  rst = n % prc; 
  displs[0] = 0;

  for(int i = 0; i < prc - 1; i++) { 
    counts[i] = (i < rst) ? m * (loc + 1) : m * loc;
    displs[i + 1] = counts[i] + displs[i];
  }

  counts[prc - 1] = m * loc;
}

void get_slices(double *A, double *B, int root, int n, int m, char *name) {
  int id, prc;
  int *counts, *displs;
  double *a, *b;

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);

  if(id == root) {
    FILE *file = fopen(name, "r");
    a = (double *)malloc(n * n * sizeof(double));
    b = (double *)malloc(n * n * sizeof(double));
    counts = (int *)malloc(prc * sizeof(int));
    displs = (int *)malloc(prc * sizeof(int));

    fscanf(file, "%d", &n);
    get_counts(counts, displs, n, n);

    for (int i = 0; i < n * n; i++) fscanf(file, "%lf", &a[i]);
    for (int i = 0; i < n * n; i++) fscanf(file, "%lf", &b[i]);

    fclose(file);
  }

  MPI_Scatterv(a, counts, displs, MPI_DOUBLE, A, n * m, MPI_DOUBLE, root, MPI_COMM_WORLD);
  MPI_Scatterv(b, counts, displs, MPI_DOUBLE, B, n * m, MPI_DOUBLE, root, MPI_COMM_WORLD);

  if(id == root) {
    free(a);
    free(b);
    free(counts);
    free(displs);
  }
}

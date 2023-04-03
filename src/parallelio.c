#include "parallelio.h"
#include "utility.h"

void generate_slices(double *A, double *B, int n, int loc) {
  int id; 

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  
  for(int i = 0; i < n * loc; i++) {
    srand(id + i + time(NULL));
    A[i] = randfrom(-1, 1);
    srand(id + i + 1 + (int)time(NULL));
    B[i] = randfrom(-1, 1);
  }
}

void distributed_print(double* A, int n, int m, int clear, char *name) {
  /*
   * Prints a 2-dimensional array of which the parts are distributed among different MPI processes as
   * vertical slices.
   * */
  int id, prc, loc, rst, cnt; 

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);
	
  loc = n / prc;
  rst = n % prc;

  if(id == 0) { // The root receives all the parts and prints them in the correct order
    FILE *file = stdout;
    double *bfr = (double *)malloc(m * n * sizeof(double));
    int flag = strcmp(name, "stdout"); // Option to print to `stdout`

    if(flag) {
      if(clear) fclose(fopen(name, "w"));
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
  else MPI_Send(A, m * n, MPI_DOUBLE, 0, id, MPI_COMM_WORLD); // Each process sends its part
}

void get_dimension(int root, int *n, char *name) {
  /*
   * Gets the dimension of the square matrix from the file called `name`, it is supposed that the 
   * dimension is the first entry of the file.
   * */
  int id, matches; 

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  
  if(id == root) {
    FILE *file = fopen(name, "r");
    matches = fscanf(file, "%d", n);
    fclose(file);
  }

  MPI_Bcast(n, 1, MPI_INT, root, MPI_COMM_WORLD);
}

void get_slices(double *A, double *B, int root, int n, int m, char *name) {
  /*
   * Reads the matries `A` and `B` to be multiplied from a file and scatters them, by dividing them
   * in vertical sliced, to each process.
   * */
  int id, prc, cnt, matches;
  double *bfr;
  FILE *file;

  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);

  if(id == root) { // The root reads the first matrix and sends parts of it
    file = fopen(name, "r");
    bfr = (double *)malloc(m * n * sizeof(double));
    
    matches = fscanf(file, "%d", &n);
    
    for(int i = 0; i < n * m; i++) matches = fscanf(file, "%lf", &A[i]);
    
    for(int i = 1; i < prc; i++) {
      cnt = (i < n % prc) ? n / prc + 1 : n / prc;
      
      for (int i = 0; i < n * cnt; i++) matches = fscanf(file, "%lf", &bfr[i]);
      
      MPI_Send(bfr, cnt * n, MPI_DOUBLE, i, i, MPI_COMM_WORLD); 
    }
  } else MPI_Recv(A, m * n, MPI_DOUBLE, root, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  if(id == root) { // The root does the same for the second matrix
    for(int i = 0; i < n * m; i++) matches = fscanf(file, "%lf", &B[i]);
    
    for(int i = 1; i < prc; i++) {
      cnt = (i < n % prc) ? n / prc + 1 : n / prc;
      
      for (int i = 0; i < n * cnt; i++) matches = fscanf(file, "%lf", &bfr[i]);
      
      MPI_Send(bfr, cnt * n, MPI_DOUBLE, i, i, MPI_COMM_WORLD); 
    }

    fclose(file);
    free(bfr);
  } else MPI_Recv(B, m * n, MPI_DOUBLE, root, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

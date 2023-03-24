#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#ifdef DGEMM
#include <cblas.h>
#endif

#include "utility.h"

int main(int argc, char** argv) {
  int n, id, prc, loc, rst, cnt, upr, root = 0; 
  int *counts, *displs;  
  double first, second, third, io_time = 0, cp_time = 0;
  double *A, *B, *C, *bfr;
  char *p, *data = "data/matrices.txt", *result = "data/result.txt", *times = "data/times.txt";
  FILE *file;
  MPI_Datatype cntgs;  

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &prc);

  n = strtol(argv[1], &p, 10);

#ifdef DEBUG 
  if(id == root) {
    file = fopen(data, "w");
    fprintf(file, "%d\n", n);
    fclose(file);
  }
#endif

  rst = n % prc;
  loc = (id < rst) ? n / prc + 1 : n / prc;
  cnt = (rst) ? n / prc + 1 : n / prc;
  upr = (rst) ? n / cnt : prc;
  
  MPI_Type_vector(loc, cnt, n, MPI_DOUBLE, &cntgs);
  MPI_Type_commit(&cntgs);
  
  counts = (int *) malloc(prc * sizeof(int));
  displs = (int *) malloc(prc * sizeof(int));
  A = (double *) malloc(n * loc * sizeof(double));
  B = (double *) malloc(n * loc * sizeof(double));
  C = calloc(n * loc, sizeof(double));
  bfr = (double *) malloc(n * cnt * sizeof(double));
  
  get_counts(counts, displs, n, cnt);
  generate_slices(A, B, n, loc);

#ifdef DEBUG
  distributed_print(A, n, loc, 0, data);
  distributed_print(B, n, loc, 0, data);
#endif // 'gg', cit. Gallo

  for(int m = 0; m < upr; m++) {
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    first = MPI_Wtime();
    
    MPI_Allgatherv(B + m * cnt, 1, cntgs, bfr, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    second = MPI_Wtime();
 
#ifdef DGEMM  
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, loc, cnt, n, 1, A, n, bfr, cnt, 0, C + m * cnt, n);
#else
    serial_multiplication(A, bfr, C + m * cnt, loc, cnt, n);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    third = MPI_Wtime();
    io_time += second - first;
    cp_time += third - second;
  }
  
  if(rst) {
    rst = n % cnt;
    
    MPI_Type_free(&cntgs);
    MPI_Type_vector(loc, rst, n, MPI_DOUBLE, &cntgs);
    MPI_Type_commit(&cntgs);

    MPI_Barrier(MPI_COMM_WORLD);
    first = MPI_Wtime();
    
    get_counts(counts, displs, n, rst);
    MPI_Allgatherv(B + upr * cnt, 1, cntgs, bfr, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    second = MPI_Wtime();
    
#ifdef DGEMM  
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, loc, rst, n, 1, A, n, bfr, rst, 0, C + upr * cnt, n);
#else
    serial_multiplication(A, bfr, C + upr * cnt, loc, rst, n);
#endif
    
    MPI_Barrier(MPI_COMM_WORLD);
    third = MPI_Wtime();
    io_time += second - first;
    cp_time += third - second;
  }
  
#ifdef DEBUG
  distributed_print(C, n, loc, 1, result);
#endif

  if(id == root) {
#ifdef DGEMM 
    char *sign = "dgemm";
#else
    char *sign = "naive";
#endif
    
    file = fopen(times, "a");
    fprintf(file, "%s %d %d %lf %lf\n", sign, n, prc, io_time, cp_time);
    fclose(file);
  }

  free(A);
  free(B);
  free(C);
  free(bfr);
  free(counts);
  free(displs);
  MPI_Type_free(&cntgs);
  MPI_Finalize();

  return 0;
}

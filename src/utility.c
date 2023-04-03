#include "utility.h"
#include "computation.h"

double randfrom(double min, double max) {
  /*
   * generate a random floating point number from min to max
   * */ 
  double range = (max - min); 
  double div = RAND_MAX / range;

  return min + (rand() / div);
}

void print(double* A, int n, int m, FILE *file) {
  /*
   * Prints a 2-dimensional array beginning at position `A` with dimensions `n` rows time `m` 
   * columns.
   * */
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) fprintf(file, "%.17g ", A[i * n + j]);

    fprintf(file, "\n");
  }
}

void test(char *data, char *result) {
  /*
   * Tests with a serial matrix multiplication if the multiplication of the matrices in `data` is 
   * equal to the matrices in `result`.
   * */
  int n, matches;
  double *A, *B, *C;

  FILE *file = fopen(data, "r");
  matches = fscanf(file, "%d", &n);

  A = (double *)malloc(n * n * sizeof(double));
  B = (double *)malloc(n * n * sizeof(double));
  C = calloc(n * n, sizeof(double));
  
  for (int i = 0; i < n * n; i++) matches = fscanf(file, "%lf", &A[i]);
  for (int i = 0; i < n * n; i++) matches = fscanf(file, "%lf", &B[i]);
  
  fclose(file);
	
  serial_multiplication(A, B, C, n, n, n);
  
  file = fopen(result, "r"); 

  for (int i = 0; i < n * n; i++) matches = fscanf(file, "%lf", &A[i]);

  fclose(file);
  
  double eps = 1e-8;
  int flag = 1;

  for (int i = 0; i < n * n; i++) {
    if(A[i] - C[i] > eps || A[i] - C[i] < -eps) {
      flag = 0;
      break;
    }
  }
  
  if(flag) {
    printf("%s\nParallel and serial results are compatible\n\n", GREEN);
  } else {
    printf("%s\nParallel and serial results are NOT compatible\n\n", RED);
  }

  printf("%s", NORMAL);

  free(A);
  free(B);
  free(C);
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define RED "\x1b[31m"
#define GREEN "\x1b[32m"
#define NORMAL "\x1b[m"

int main() {
  int n;
  int *counts, *displs;
  double *A, *B, *C;
  char *data = "data/matrices.txt", *result = "data/result.txt", *times = "data/times.txt";

  FILE *file = fopen(data, "r");
  fscanf(file, "%d", &n);
	
  A = (double *)malloc(n * n * sizeof(double));
  B = (double *)malloc(n * n * sizeof(double));
  C = calloc(n * n, sizeof(double));

  clock_t first = clock();
  
  for (int i = 0; i < n * n; i++) fscanf(file, "%lf", &A[i]);
  for (int i = 0; i < n * n; i++) fscanf(file, "%lf", &B[i]);
	
  fclose(file);
  clock_t second = clock();
	
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
      for(int k = 0; k < n; k++)
        C[i * n + j] += A[i* n + k] * B[k * n + j];
  
  file = fopen("data/result.txt", "r");
 
  for (int i = 0; i < n * n; i++) fscanf(file, "%lf", &A[i]);

#ifdef DEBUG
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) printf("%lf ", C[i * n + j]);

    printf("\n");
  }
#endif

  fclose(file);
  
  double eps = 1e-8;
  int flag = 1;
  
  for (int i = 0; i < n * n; i++) {
    if(A[i] - C[i] > eps || A[i] - C[i] < -eps) {
      printf("%d %f\n", i, A[i] - C[i]);
      flag = 0;
      break;
    }
  }
  
  if(flag) {
    printf("%sParallel and serial results are compatible\n", GREEN);
  } else {
    printf("%sParallel and serial results are NOT compatible\n", RED);
  }

  printf("%s", NORMAL);

  free(A);
  free(B);
  free(C);
}

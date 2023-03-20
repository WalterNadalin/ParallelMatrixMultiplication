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
  char *data = "data/matrices.txt", *result = "data/serial_result.txt";

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


  clock_t third = clock();
  fclose(fopen(result, "w"));
  file = fopen(result, "a");
  
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) fprintf(file, "%lf ", C[i * n + j]);

    fprintf(file, "\n");
  }
  
  clock_t io = (clock() - third) + (second - first);
  clock_t cp = third - second;
  double io_time = (double)io / CLOCKS_PER_SEC;
  double cp_time = (double)cp / CLOCKS_PER_SEC;
  printf("\nSerial read and write:\t %f [s]\nSerial computation:\t %f [s]\n", io_time, cp_time);
  
  fclose(file);
  file = fopen("data/result.txt", "r");
 
  for (int i = 0; i < n * n; i++) fscanf(file, "%lf", &A[i]);
  
  fclose(file);
  
  int flag = 1;
  
  for (int i = 0; i < n * n; i++) if(abs(A[i] - C[i]) > 1E-7) {
    flag = 0;
    break;
  }
  
  if(flag) {
    printf("\n%sParallel and serial results are compatible\n", GREEN);
  } else {
    printf("\n%sParallel and serial results are NOT compatible\n", RED);
  }

  printf("%s", NORMAL);

  free(A);
  free(B);
  free(C);
}

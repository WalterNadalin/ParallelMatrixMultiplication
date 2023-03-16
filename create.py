import sys
from random import random

n = int(sys.argv[1])
A = [random() for _ in range(n * n)]
B = [random() for _ in range(n * n)] 
C = [0. for _ in range(n * n)]

for i in range(n):
  for j in range(n):
    for k in range(n):
      C[i * n + j] += A[i * n + k] * B[k * n + j]

with open('data/matrices.txt', 'w') as text:
  text.write(str(n) + '\n')
  text.write(' '.join(str(value) for value in A) + '\n')
  text.write(' '.join(str(value) for value in B) + '\n') 

with open('data/result.txt', 'w') as text:
  text.write(' '.join(str(value) for value in C) + '\n')

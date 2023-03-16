import sys
from random import random

n = int(sys.argv[1])
A = [random() for _ in range(n * n)]
B = [random() for _ in range(n * n)] 

with open('data/matrices.txt', 'w') as text:
  text.write(str(n) + '\n')
  text.write(' '.join(str(value) for value in A) + '\n')
  text.write(' '.join(str(value) for value in B) + '\n') 

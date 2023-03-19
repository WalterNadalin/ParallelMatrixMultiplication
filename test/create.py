import sys
from random import random

n = int(sys.argv[1])
A = [[random() for _ in range(n)] for _ in range(n)]
B = [[random() for _ in range(n)] for _ in range(n)] 

with open('data/matrices.txt', 'w') as text:
  text.write(str(n) + '\n')

  for line in A:
    text.write(' '.join(str(value) for value in line) + '\n')
  
  text.write('\n');

  for line in B:
    text.write(' '.join(str(value) for value in line) + '\n') 

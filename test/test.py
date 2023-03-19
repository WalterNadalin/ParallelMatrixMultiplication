from math import isclose

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

with open('data/matrices.txt', 'r') as text:
  n = int(next(text))
  A = [[float(x) for x in next(text).split()] for _ in range(n)]
  next(text)
  B = [[float(x) for x in next(text).split()] for _ in range(n)]
  C = [0. for _ in range(n * n)]

for i in range(n):
  for j in range(n):
    for k in range(n):
      C[i * n + j] += A[i][k] * B[k][j]

with open('data/result.txt', 'r') as text:
  R = [float(value) for line in text for value in line.split()]

flag = 0

for c, r in zip(C, R):
    if not isclose(round(c, 6), r):
        flag = 1

if flag:
  print(f"{bcolors.FAIL}Test failed: the result is not the correct one.{bcolors.ENDC}")
else:
  print(f"{bcolors.OKGREEN}Test passed: the result is correct.{bcolors.ENDC}")

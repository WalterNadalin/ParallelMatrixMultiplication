# Parallel matrix multiplication
Given two $n$-dimensional square matrices $A, B \in\mathcal{M}_{n}(\mathbb{R})$ defined on the real field, it is know that the matrix multiplication $C=AB\in\mathcal{M}_n(\mathbb{R})$ is defined as

$$
[AB]_{i, j}=\sum_{r=1}^n a_{i,r}b_{r, j},
$$

and, a naive approach to implement such an operation in the programming language `C` (in which multidimensional arrays are stored row-major order), is the following:

```C
for(i = 0; i < n; i++)
  for(j = 0; j < n; j++)
    for(k = 0; k < n; k++)
      C[i][j] += A[i][k] + B[k][j];
 ```
 
which is an example of a *serial* code (implemented in `test\serial_multiplication.c`), that is a code which run on only one computation unit.
 
Here, my goal is to implement a parallel code in `C` to perform such an operation given any number $m$ of computational units. 

To compile and run one should install a library which implements the MPI standard (for example https://www.open-mpi.org/ or https://www.mpich.org/) and the `mpicc` wrapper for `gcc`, in particular:
- to compile you can use the command `make` in the main directory
- to run there are two bash scripts which allows you simply to run or also to test (that is, checking the result with a serial implementation) the parallel implementation, respectively they are `scripts\run.sh` and `scripts\test.h`. Both of them require at least one parameter, the number of processors, and you can also give the dimension of the square matrices of which to compute the multiplication (which are generated randomly and contained in the file `data\matrices.txt`). For example, to generate two random $n\times n$ matrices and to run the program with $m$ processes you should type
  ```bash
  bash .\scripts\run.sh n m
  ```
  or, if you want only to run the program on $m$ processes without generating any matrix, use
  ```bash
  bash .\scripts\run.sh m
  ```
  and analogously for the other script. The result will be contained in the file `data\result.txt`.
  
If you want to give your own matrices you should write them in the file `data\matrices.txt` and also write in it (before them) their dimension.

 
These are the things done or to be done:
1. Implement a working code using only MPI
- [x] Implement a working code when $n$ multiple of $m$
- [x] Add feature to read matrices from file
- [x] Add some testing
- [x] Implement a working code when $n$ generic 
- [x] Measure times
- [ ] ...
2. ...

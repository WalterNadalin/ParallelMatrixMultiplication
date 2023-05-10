# Parallel matrix multiplication

Given two $n$-dimensional square matrices $A, B \in\mathcal{M}_{n}(\mathbb{R})$ defined on the real field, it is know that the matrix multiplication $C=AB\in\mathcal{M}_n(\mathbb{R})$ is defined as

$$
\[AB\]\_{i, j}\=\sum\_{r=1}^na_{i,r}b_{r, j}
$$

and, a naive approach to implement such an operation in the programming language `C` (in which multidimensional arrays are stored row-major order), is the following:

```C
for(i = 0; i < n; i++)
  for(j = 0; j < n; j++)
    for(k = 0; k < n; k++)
      C[i][j] += A[i][k] + B[k][j];
 ```
 
which is an example of a *serial* code.
 
Here, my goal is to implement a parallel code in `C` to perform such an operation given any number $m$ of computational units. 

**Important note**: the code has been written to compile and run on the *Marconi100* cluster at **CINECA**. To compile and run the `Spectum_MPI`, the `cuda` and the `openablas` modules must be loaded. 

### Compilation
---
To **compile** it is possible to use the command 
```
make [version] [debug=yes]
``` 
where `[version]` can be either blank, `dgemm` or `cuda`. This will produce the `[version]multiplication.x` executable (the name will depend on which version it has been compiled). 

### Execution
---
The executable will generate two $n\times n$ matrices with random entries and multiply them. To **run**, for example with 3 processes using $16\times 16$ matrices, it is possible either to use `mpirun -np 3 ./[version]multiplication.x 16` or to use
```bash
make [version]run prc=3 dim=16 [debug=yes]
```
where `[version]` can be either blank, `dgemm` or `cuda`. This will produce the `[version]multiplication.x` executable (the name will depend on how it has been compiled) and run it. 

### Test
---
To **test** it is possible either to pass the `debug=yes` flag to the `Makefile`
```bash
make [version] debug=yes
```
and then run the `[version]debug_multiplication.x` executable using `mpirun`. It is also supported the command:
```bash
make [version]run debug=yes
```
where `[version]` can be either blank, `dgemm` or `cuda`. This will compile (if necessary) and run immediately after.


## To do list
These are the things done or to be done:
1. Implement a working code using only MPI
- [x] Implement a working code when $n$ multiple of $m$
- [x] Add some testing
- [x] Implement a working code when $n$ generic 
- [x] Measure performances
2. Include a version using `cblas_dgemm` instead of the serial multiplication done by each MPI process
- [x] Make it work
- [x] Make some plots to compare performances with serial version
3. Port on GPU 
- [x] Include a version using `cublasDgemm` instead of the serial multiplication done by each MPI process
- [x] Make some plots to compare performances with serial and `cblas_dgemm` versions

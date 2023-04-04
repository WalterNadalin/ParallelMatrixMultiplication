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
 
which is an example of a *serial* code.
 
Here, my goal is to implement a parallel code in `C` to perform such an operation given any number $m$ of computational units. 

**Important note**: the code has been written to compile and run on the *Marconi100* cluster at **CINECA**.

To compile and run a library which implements the `Spectum_MPI` and the `cuda` modules must be loaded. In particular, the `Makefile` assumes wrapper `mpicc` and the compiler `nvcc` are available.

### Compilation
---
To **compile** it is possible to use the command 
```
make [version]
``` 
where `[version]` can be either blank, `dgemm` or `cuda`. This will produce the `multiplication.x` executable. 

### Execution
---
The executable will generate two $n\times n$ matrices with random entries and multiply them. To **run**, for example with 3 processes using $16\times 16$ matrices, it is possible either to use the command 
```bash
mpirun -np 3 ./multiplication.x 16
```
or to, more conveniently, use the `./scripts/run.sh` bash scripts. It recquires 2 parameters:
- the number $m$ of processes,
- the size $n$ of the square matrices.
  
For example
```bash
bash ./scripts/run.sh 3 16 [version]
```
where `[version]` can be either blank, `dgemm` or `cuda`. The program will generate two random $16\times 16$ matrices and will run the program with $3$ processes. 

### Testing and debugging
---
To **test** it is possible either to pass the `flag=debug` variable to the `Makefile`
```bash
make [version] flag=debug
```
and then run the `multiplication.x` executable or to compile and run using the script with the `debug` parameter
```bash
bash ./scripts/run.sh 3 16 [version] debug
```
where `[version]` can be either blank, `dgemm` or `cuda`. This will make the program write the matrices generated in the file `data/matrices.txt` and the resulting one in `result.txt`. Then the program will check if the result written is compatible with the one obtained with a serial implementation of the multiplication.

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
- [ ] Implement a working matrix multiplication in `Cuda` og a GPU device
- [ ] ...

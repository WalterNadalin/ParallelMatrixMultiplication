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

To compile and run a library which implements the MPI standard is required, the script also assumes that the wrapper `mpicc` is available.

### Compilation
---
to **compile** it is possible to use the command 
```
make
``` 
in the main directory, it will produce the `multiplication.x` executable,

### Execution
---
The executable will generate two $n\times n$ matrices with random entries and multiply them. To **run**, for example with 3 processes using $16\times 16$ matrices, it is possible to use the command 
```bash
mpirun -np 3 ./multiplication.x 16
```
or to, more conveniently, use a bash scripts. It recquires 2 parameters:
- the number $m$ of processes,
- the size $n$ of the square matrices.
  
For example
```bash
bash .\scripts\run.sh 3 16
```
will generate two random $16\times 16$ matrices and will run the program with $3$ processes,

### Testing and debugging
---
To **test** it is necessary to compile with the `debug` paramater
```bash
bash .\scripts\run.sh 3 16 debug
```
this will make the program write the matrices generated in the file `data/matrices.txt` and the resulting one in `result.txt`. Then the program will check if the result written is compatible with the one obtained with a serial implementation of the multiplication.

## To do list
These are the things done or to be done:
1. Implement a working code using only MPI
- [x] Implement a working code when $n$ multiple of $m$
- [x] Add some testing
- [x] Implement a working code when $n$ generic 
- [x] Measure performances
2. Include a version using `cblas_dgemm` instead of the serial multiplication
- [x] Make it work
- [x] Make some plots to compare performances with serial version
3. Port on GPU 
- [x] Include a version using `cublasDgemm` instead of the serial multiplication
- [x] Make some plots to compare performances with serial and `cblas_dgemm` version
- [ ] Implement a working matrix multiplication in `Cuda` og a GPU device
- [ ] ...

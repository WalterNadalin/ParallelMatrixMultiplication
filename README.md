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
 
which is an example of a *serial* code, that is a code which run on only one computation unit.
 
Here, my goal is to implement a parallel code in `C` to perform such an operation given any number $m$ of computational units.
 
These are the things done or to be done:
 
- [x] Implement a working code using only MPI
  - [x] Implement a working code when $n$ multiple of $m$
    - [x] Add feature to read matrices from file
    - [ ] Add some testing
  - [ ] Implement a working code when $n$ generic 

# Chapter 3 - Multidimensional grids and data

## Exercises

#### 1. In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.

#### a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.

```c++
__global__
void matrixMulRowKernel(float* M, float* N, float* P, int row_M, int col_M, int row_N, int col_N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < row_M){
        for (int c=0; c<col_N; c++){
            float P_value = 0;
            for (int k=0; k<col_M; k++){
                P_value += M[row*col_M+k] * N[k*col_N+c];
            }
            P[row*col_N + c] = P_value;
        }
    }
}
```

#### b. Write a kernel that ahs each thread produce one output matrix column. Fill in the execution configuration parameters for the design.

```c++
__global__
void matrixMulColKernel(float* M, float* N, float* P, int row_M, int col_M, int row_N, int col_N){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < col_N){
        for (int r=0; r<row_M; r++){
            float P_value = 0;
            for (int k=0; k<col_M; k++){
                P_value += M[r*col_M+k] * N[k*col_N+col];
            }
            P[r*col_N + col] = P_value;
        }
   }
}
```

#### c. Analyze the pros and cons of each of the two kernel designs.

Both row-wise and column-wise kernels result in fewer thread launch (**Pros**). But on the other hand, each thread carries more computational burden (**Cons**). 

In terms of memory access, for both row-wise and column-wise kernels, memory access for N seems to be inefficient, as a thread has to jump from one memory location to another.
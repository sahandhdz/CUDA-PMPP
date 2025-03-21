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

#### 2. A matrix-vector multiplication takes an input matrix B and a vector c and produces one output vector A. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: pointer to the input matrix, pointer to the input vector, pointer to the output vector. Use one thread to calculate an output vector element.

```c++
// Kernel is designed for square matrices (n_cols = n_rows = n_dim)
__global__
void matrixVecMulKernel(float* B, float* V, float*A, size_t n_dim){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < n_dim){
        float a_value = 0;
        for (int k=0; k<n_dim; k++){
            a_value += B[row*n_dim + k] * V[k];
        }
        A[row] = a_value;
    }
}


void matrixVecMul(float* B_h, float* V_h, float*A_h, size_t n_dim){
    float* B_d;
    float* V_d;
    float* A_d;

    size_t vec_size = n_dim*sizeof(float);

    cudaMalloc((void**)&B_d, vec_size*n_dim); // B_d should be allocated to store matrix with n_dim^2 float elements
    cudaMalloc((void**)&V_d, vec_size);
    cudaMalloc((void**)&A_d, vec_size);

    cudaMemcpy(B_d, B_h, vec_size*n_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V_h, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A_h, vec_size, cudaMemcpyHostToDevice);

    dim3 dimgrid(ceil(n_dim/32.0), 1, 1);
    dim3 dimblock(32, 1, 1);
    matrixVecMulKernel<<<dimgrid, dimblock>>>(B_d, V_d, A_d, n_dim);

    cudaMemcpy(A_h, A_d, vec_size, cudaMemcpyDeviceToHost);

    // Dont forget to free the allocated memory
    cudaFree(B_d);
    cudaFree(V_d);
    cudaFree(A_d);
}
```

It should be noted that the kernel is designed for square matrices.

#### 3. Consider the following CUDA kernel and the corresponding host function that calls it:

```c++
__global__
void foo_kernel(float *a, float *b, unsigned int M, insigned int N){
    unsigned int row = blockIdx.y*blockDim.y + threadIDx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIDx.x;
    if (row < M && col < N){
        b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
    }
}

void foo(float *a_d, float *b_d){
    unsigned int M = 150;
    unsigned int N = 300;
    dim3 bd(16, 32);
    dim3 gd((N-1)/16 + 1, (M-1)/32 + 1);
    foo_kenel<<<gd, bd>>>(a_d, b_d, M, N);
}
```

#### a. What is the number of threads per block?

The number of threads per block is `16*32=512`.

#### b. What is the number of threads in the grid?

Grid is 19 by 5. So, in total, we have `19 * 5 * 512 = 48640` threads

#### c. What is the number of blocks on the grid?

The number of blocks in the grid is `19*5=95`.

#### d. What is the n umber of threads that execute the code on line 6?

Only `150*300=45000` threads pass the condition at line 5 and proceeds to line 6.

#### 4. Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one dimensional array. Specify the array index of the matrix element at row 20 and column 10:

#### a. If the matrix is stored in row-major order.

Assuming 0-based indexing, for the element at 20th column and 10th row, the index is `(20-1)*400 + (10-1) = 7609`.

#### b. If the matrix is stored in column-major order.

Assuming 0-based indexing, for the element at 20th column and 10th row, the index is `(10-1)*500 + (20-1) = 4519`.

#### 5. Consider a 3D tensor with width of 400, height of 500, and depth of 300. The tensor stored in a one-dimensional array in a row-major order. Specify the index of the tensor at x=10, y=20, z=5.

Assuming the 0-based indexing, the index of the element should be `(5-1)*(400*500)+(10-1)*400+(20-1)`.


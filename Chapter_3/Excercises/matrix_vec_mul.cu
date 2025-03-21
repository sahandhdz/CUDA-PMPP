#include <stdio.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

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

int main(){
    size_t n_dim =3;
    float B[3][3] = {{1.0, 4.0, 2.0}, {3.0, 5.0, 8.0}, {9.0, 2.0, 1.0}};
    float V[3] = {3.0, -1.0, 2.0};
    float A[3] = {0.0, 2.0, 0.0};

    matrixVecMul(&B[0][0], V, A, n_dim);

    for (size_t i=0; i<n_dim; i++){
        std::cout << A[i] << ", ";
    }
    std::cout << std::endl;
}
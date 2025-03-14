#include <stdio.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i<n){
        C[i] = A[i] + B[i];
    }
}





void vecAdd(float* A_h, float *B_h, float *C_h, int n){
    int size = n* sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaError_t err;

    // Part 1: Allocate device memory for A, B, and C
    // Copy A and B to device memory
    err = cudaMalloc((void**)&A_d, size);
    if (err == cudaSuccess){
        printf("Successful device memory allocation!\n");
    }else{
        printf("Unsuccessful memory allocation!\n");
    }

    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Part 2.1: Copy from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);


    // Part 2.3: Copy from device to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);


    // Part 3: Free the allocated memory on teh device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){

    float a[] = {1,2,3};
    float b[] = {3,4,5};

    float c[] = {0.0,0.0,0.0};

    int N = 3;


    vecAdd(a, b, c, N);

    for (int i=0; i<N; i++){
        printf("number: %f\n", c[i]);
    }

    return 0;
    
}
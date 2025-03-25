#include <stdio.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#define TILE_WIDTH 16


// A better more efficient version of the matrix multiplication kernel
// Improved by using shared memory
__global__
void matrixMulKernel(float* M, float* N, float* P, int width){
    // Shared memory, accessible for all threads inside a block.
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // The row amd column of P to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH * tx;

    // Loop over tiles
    float Pvalue = 0;
    for (int ph=0; ph<width/TILE_WIDTH; ph++){
        // collaborative loading of M and N tiles into shared memory
        Mds[ty][tx] = M[row*width+ph*TILE_WIDTH+tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH+ty)*width + col];
        __syncthreads();

        for (int k =0; k<TILE_WIDTH; k++){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[row*width + col] = Pvalue;
}

void matrixMul(float* M_h, float* N_h, float* P_h, int width){
    float* M_d;
    float* N_d;
    float* P_d;

    size_t matrix_size = width*width*sizeof(float);

    cudaMalloc((void**)&M_d, matrix_size);
    cudaMalloc((void**)&N_d, matrix_size);
    cudaMalloc((void**)&P_d, matrix_size);

    cudaMemcpy(M_d, M_h, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(P_d, P_h, matrix_size, cudaMemcpyHostToDevice);

    dim3 dimgrid(ceil(width/32.0), ceil(width/32.0), 1);
    dim3 dimblock(32, 32, 1);

    matrixMulKernel<<<dimgrid, dimblock>>>(M_d, N_d, P_d, width);

    cudaMemcpy(P_h, P_d, matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

void print_matrix(float* A, int row, int col){
    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++){
            std::cout << *(A + i*col + j) << " ";
        }
        std::cout << std::endl;
    }
}

int main(){
    int width = 4;
    float M[4][4] = {{3.0, 2.0, 1.0, 2.0}, {4.0, -1.0, 1.0, 3.0}, {5.0, 2.0, 3.0, 4.0}, {7.0, -3.0, -1.0, 3.0}};

    float N[4][4] = {{1.0, 4.0, -2.0, 1.0}, {3.0, -1.0, 5.0, 4.0}, {7.0, 3.0, 2.0, 3.0}, {3.0, 2.0, 4.0, 5.0}};

    float P[4][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};

    matrixMul(&M[0][0], &N[0][0], &P[0][0], width);

    print_matrix(&P[0][0], width, width);
}
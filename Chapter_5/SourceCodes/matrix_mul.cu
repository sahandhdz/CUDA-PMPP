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

        for (int k =0; k<TILE_WIDTH k++){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[row*width + col] = Pvalue;
}
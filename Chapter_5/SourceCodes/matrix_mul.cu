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


}
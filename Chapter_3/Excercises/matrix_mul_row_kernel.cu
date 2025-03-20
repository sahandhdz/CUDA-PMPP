#include <stdio.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>


__global__
void matrixMulRowKernel(float* M, float* N, float* P, int row_M, int col_M, int row_N, int col_N){
    int row = blockIdx.y * blockDim.y + threadIdx. y;

    if (row < row_M){
        // float P_value = 0;
        // for (int k=0; k<col_M; k++){
        //     P_value += M[row*col_M+k] * N[k*col_N + col]; 
        // }
        // P[row*col_N + col] = P_value;
        for (int c=0; c<col_N; c++){
            float P_value = 0;
            for (int k=0; k<col_M; k++){
                P_value += M[row*col_M+k] * N[k*col_N+c];
            }
            P[row*col_N + c] = P_value;
        }
    }
}
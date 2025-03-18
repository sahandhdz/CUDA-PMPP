#include <stdio.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>


__global__
void matrixMulKernel(float* M, float* N, float* P, int row_M, int col_M, int row_N, int col_N){
    int row = blockIdx.y * blockDim.y + threadIdx. y;
    int col = blockIdx.x * blockDim.x + threadIdx. x;

    if ((row < row_M) && (col < col_N)){
        float P_value = 0;
        for (int k=0; k<col_M; k++){
            P_value += M[row*col_M+k] * N[k*col_N + col]; 
        }
        P[row*col_N + col] = P_value;
    }
}

void matrixMul(float* M_h, float* N_h, float* P_h, int row_M, int col_M, int row_N, int col_N){

    if (col_M != row_N){
        std::cerr << "Error: Invalid Matrix Dimensions. col_M = " << col_M << " must be qual to row_N = " << row_N << std::endl;
        return;
    }

    int size_M = row_M*col_M*sizeof(float);
    int size_N = row_N*col_N*sizeof(float);
    int size_P = row_M*col_N*sizeof(float);

    float* M_d;
    float* N_d;
    float* P_d;

    cudaMalloc((void**)&M_d, size_M);
    cudaMalloc((void**)&N_d, size_N);
    cudaMalloc((void**)&P_d, size_P);

    cudaMemcpy(M_d, M_h, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(P_d, P_h, size_P, cudaMemcpyHostToDevice);

    dim3 dimgrid(ceil(col_N/32.0), ceil(row_M/32.0), 1);
    dim3 dimblock(32, 32, 1);

    matrixMulKernel<<<dimgrid, dimblock>>>(M_d, N_d, P_d, row_M, col_M, row_N, col_N);

    cudaMemcpy(M_h, M_d, size_M, cudaMemcpyDeviceToHost);
    cudaMemcpy(N_h, N_d, size_N, cudaMemcpyDeviceToHost);
    cudaMemcpy(P_h, P_d, size_P, cudaMemcpyDeviceToHost);

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

    int M_rows = 3;
    int M_cols = 2;
    float M[3][2] = {{3.0, 2.0}, {4.0, -1.0}, {11.0, 9.0}};

    int N_rows = 2;
    int N_cols = 3;
    float N[2][3] = {{1.0, 4.0, -2.0}, {3.0, -1.0, 5.0}};

    float P[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    matrixMul(&M[0][0], &N[0][0], &P[0][0], M_rows, M_cols, N_rows, N_cols);

    print_matrix(&P[0][0], 3, 3);

}
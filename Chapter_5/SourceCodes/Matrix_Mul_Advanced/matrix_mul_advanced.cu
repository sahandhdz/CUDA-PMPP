#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define TILE_WIDTH 2


__global__
void matrixMulKernel(float* M, float* N, float* P, int width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int ph = 0; ph<width/TILE_WIDTH+1; ph++){
        Mds[ty][tx] = M[row*width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ty+ph*TILE_WIDTH)*width + col];
        __syncthreads();

        for (int k=0; k<TILE_WIDTH; k++){
            Pvalue += Mds[ty][k]*Nds[k][tx];
        }
        __syncthreads();
    }
    P[row*width+col] = Pvalue;
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

    // width of the matrices in this implementation should be divisible by width of a thread block
    dim3 dimgrid(ceil(width/2.0), ceil(width/2.0), 1);
    dim3 dimblock(2, 2, 1);

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
    // int width = 4;
    // float M[4][4] = {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}};

    // float N[4][4] = {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}};

    // float P[4][4] = {{4.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};

    int width = 5;
    float M[5][5] = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    float N[5][5] = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    float P[5][5] = {{4.0, 0.0, 0.0, 0.0, 0}, {0.0, 0.0, 0.0, 0.0, 0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}};



    matrixMul(&M[0][0], &N[0][0], &P[0][0], width);

    print_matrix(&P[0][0], width, width);
}
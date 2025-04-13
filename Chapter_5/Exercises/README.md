# Chapter 5 - Memory architecture and data locality

## Exercises

#### 1. Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.

Using shared memory in this case does not help, and may even result in a slower performance. In the element-wise addition of two given matrices A and B, resulting in C, if we assume that each thread calculate one element of the matrix C, two different threads use different elements from A and B, so there are no shared usage between the threads and as a result no chance for cooperation, in contrast to what we saw in the case of matrix multiplication.

#### 2. Draw the equivalent of Fig. 5.7 for a 8 x 8 matrix multiplication with 2 x 2 tiling and 4 x 4 tiling. Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.

You can think like this. For the case of 8 x 8 matrices, with a tiling of size 1 (what we saw in chapter 2 for example), each thread have to read 8 elements of a row of matrix A and 8 elements of a column of matrix B. So, in total 16 readings from global memory is required.

Now let increase tiling size to 2. In this case, the 2 x 2 block (4 threads) should compute 4 elements of the output matrix. To do so, 2 rows from A and 2 columns from be have to be read from global memory (in a collaborative way). So, in total there will be 32 reads from global memory. Accordingly, there will be `32/4=8` reads per threads. Compared to the case of 1 x 1 tile, we see %50 less reads.

Now moving to the case of 4 x 4 tiling, we can see that 64 readings from global memory is needed (4 rows of matrix A and 4 columns of matrix B). So, there will be `64/16=4` readings per thread (16 is the number of total treads in the block/tile). It confirms that the number of total accesses to global memory has been divided by 4

#### 3. What type of incorrect execution behavior can happen if one forgot to use one or both `__syncthreads()` in the kernel of Fig 5.9?

The two `_syncthreads()` commands used in this kernel are used to make sure that the collaborative work of reading from global memory and writing into the shared memory are being done correctly. If we omit the  the first thread synchronization action, the process of reading all required data from global memory may not complete on time, so the threads proceed to calculate the `Pvalue` without the required data which obviously can lead to incorrect calculations. 

The second synchronization action, actually makes sure that all the required calculations with the available data on the shared memory are done and now the threads can move on to the next step, which is reading another tile of float numbers from global memory **or** writing the final value into the the output matrix. If we omit this synchronization, some threads may start transferring data from global memory to the shared memory while some other threads are still working on the previous data. This will eventuate in erroneous calculations.

#### 4. Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values etched from global memory? Explain your answer.

The main point is that the a register is private to a single thread while the scope of access for the shared memory is all the threads inside a block. Accordingly, if data can be used by more than one thread, it can be collaboratively read from the global memory and be written to the shared memory, decreasing the volume of reading data while also saving global memory bandwidth.

#### 5. For our tile matrix-matrix multiplication kernel, if we use a 32 x 32 tile, what is the reduction of memory bandwidth for input matrices M and N?

The total accesses to the global memory for reading the stored elements of matrices A and B will be reduced by a factor of 32.

#### 6. Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?

Local variables are scoped into each single thread. So, for each thread an exclusive version of that variable will be created. Hence, in total we will have `512,000` versions of that variable.

#### 7. In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?

Shared memories are scoped into each block, i.e., for each block of threads one version will be created. Hence, 1000 versions of that variable will be created through the lifetime execution of the kernel.

#### 8. Consider performing a matrix multiplication of two input matrices with dimensions N x N. How many times is each element in the input matrices requested from global memory when:

#### a. There is no tiling.

Each element will be read `N` times.

#### b. Tiles of size T x T are used?

As we discussed in previous questions, for a tile of size T, the total number of memory accesses will be decreased by a factor of T. SO each element from the input matrices will be read for `N/T` times.

#### 9. A kernel performs 36 floating-point operations and seven 32-bit global memory access per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory bound.

#### a. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second.

#### b. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second.

#### 10. To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.

```c++
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDIm(A_width/blockDim.x, A_heigh/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_with, A_height);

__global__
void BlockTranspose(float* A_elements, int A_width, int A_height){
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];
    int baseIdx - blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;
    
    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
    
    A_elements[baseOdx] = blockA[threadIdx.x][threadIdx.y];
}
```



#### 11. Consider the following CUDA kernel and the coresponding host function that calls it:

```c++
__global__ void foo_kernel(float* a, float* b){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float x[4];
    __shared__ float y_s;
    __shared__ float b_s[128];
    for (unsigned int j=0; i<4; j++){
        x[j] = a[j*blockDim.x*gridDim.x + i];
    }
    if (threadIdx.x == 0){
        y_s = 7.4f;
    }
    b_s[threadIdx.x] = b[i];
    __syncthreads();
    b[i] = 2.5*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3] + y_s*b_s[threadIdx.x] + 
        b_s[(threadIdx.x + 3)%128];
}

void foo(int* a_d, int* b_d){
    unsigned int N = 1024;
    foo_kernel<<<(N+128-1)/128, 128>>>(a_d, b_d);
}
```

#### a. How many versions of the variable i are there?

There exists 8 blocks, with 128 threads in each block, resulting in 1024 threads in total. Variable `i` has a thread scope, meaning that it is specific to each thread. So there will be 1024 versions of variable `i`. **The location where this variable is stored is the register**.

#### b. How many versions of the array x[] are there?



#### c. How many versions of the variable y_s are there?

#### d. How many versions of the array b_s[] are there?
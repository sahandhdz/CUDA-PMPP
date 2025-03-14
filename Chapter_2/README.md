# Chapter 2 - Heterogenous data and parallel computing

## Exercises

#### 1. If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?

CUDA organizes thread execution in three levels:

1. **Grid**: contains blocks.
2. **Blocks**: contains threads.
3. **Thread**: the smallest execution unit.

Each thread has a unique index, determined by:

1. `blockIdx.x` : Block index within the grid.
2. `blockDim.x` : The total number of threads within the grid.
3. `trheadIdx.x` : The index of a thread, within a block.

Accordingly, the code snippet below cane be used to uniquely identify each thread globally across all blocks:

```c++
int i =  blockIdx.x * blockDim.x + threadIdx.x;
```

We can then use this global index to map threads to data indexes.

**Note**: These are built-in variables in CUDA that help define thread hierarchy. They come from `<device_launch_parameters.h>` header, so don't forget to include!



#### 2. Assume that we want to use each thread to calculate two adjacent elements of  a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to accessed by a thread?

In this case, the number of required threads should ne halved. So we can write the kernel function as:

```c++
__gloabl__
void vecAddKernel(float* A, float* B, float* C, int n){
    // Answer to the question is the next line of code!
    int i = (blockIdx.x * blockDim.x + threadIdx.x)*2; //
    if (i<n){
        C[i] = A[i] + B[i];
    }
    if (i+1<n){
        C[i] = A[i] + B[i];
    }
}
```

The global thread index is defined by the expression `blockIdx.x + blockDim.x + threadIdx.x`. So, for example, thread 0 performs the calculation for c[0] and c[1], thread 1 performs the calculations for c[2] and c[3] and this process continues till the end.

#### 3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes `2*blockDim.x` consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable (i) should be the index for first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?

Assume that each block has 4 threads (`blockDim.x = 4`), and we have a total 16 elements to be processed. Each block will process two sections of size 4, so two blocks are required. Block 0, first process elements {0, 1, 2, 3} (section1), then elements {4, 5, 6, 7} (section 2). Block 1, similarly, first processes elements {8, 9, 10, 11} (section 1) and then {12, 13, 14, 15} (section 2). Accordingly, we come up with the implementation below:

```c++
__gloabl__
void vecAddKernel(float* A, float* B, float* C, int n){
    // Answer to the question is the next line of code!
	int i = (2 * blockIdx.x * blockDim.x) + threadIdx.x; // mapping
    if (i<n){
        C[i] = A[i] + B[i];
    }
    int j = i + blockDim.x;
    if (j<n){
        C[i] = A[i] + B[i];
    }
}
```

#### 4. For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?

Assuming the kernel size of 1024, the call to the kernel form the host will be like this:

```c++
vecAddKernel<<<ceil(n/1024), 1024>>>(A_d, B_d, C_d, n);
```

As `n=8000`, we need `ceil(n/1024) = 8` kernels, resulting in `1024*8 = 8192` threads in the grid.

#### 5. If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the `cudaMalloc` call?

The second argument of the `cudamalloc` call determines the size of the data. So it should be `v * sizeof(int)`.  

#### 6. If we want to allocate an array of n floating point elements and have a floating point pointer variable A_d to point to the allocated memory, what would be an appropriate expression for the first argument of `cudaMalloc()` call?

The first argument should be `(void **) &A_d`. 

Well, it might be a little hard to understand. So consider the below facts:

1. The first argument should be a pointer to a void pointer.
2. A_d is pointer to a float.
3. &A_d will be of type `float **`.
4. `(void **) &A_d` casts the **pointer to a float pointer** to the **pointer to a void pointer**.

#### 7. If we want to copy 3000 bytes of data from host array `A_h` to device array `A_d`, what would be an appropriate API call for this data copy in CUDA.

```c++
cudaMemcpy(A_d, A_h, 3000, cudaMemcopyHostToDevice);
```

#### 8. How would one declare a variable err that can appropriately receive the returned value of a CUDA API call?

You can use built-in CUDA error type `cudaError_t` for this purpose. The implementation should be something like this:

```c++
cudaError_t err;
err = cudaMalloc((void**)&A_d, size);
if (err == cudaSuccess){
	printf("Successful device memory allocation!\n");
}else{
	printf("Unsuccessful memory allocation!\n");
}
```

#### 9. Consider the following CUDA kernel and the corresponding host function that calls it:
```c++
__global__
void foo_kernel(float* a, float* b, unsigned int N){
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<N){
		b[i] = 2.7f*a[i] - 4.3f;
	}
}

void foo(float* a_d, float* b_d){
	unsigned int N = 200000;
	foo_kernel<<<(N+128-1)/128, 128>>>(a_d, b_d, N);
}
```

#### What is the number of threads per block?

128

#### What is the number of threads in the grid?

We have 200127/128=1563 blocks. So there will be 1563*128=200,064 threads in total.

#### What is the number of blocks in the grid?

As we said above, 1563 blocks.

#### What is the number of threads that execute the code on line 03?

All threads execute this like, so 200,064 threads execute this line in parallel.

#### What is the number of threads that execute the code on line 05?

Only the first 200,000 threads pass the condition on line 04. Hence, 200,000 threads execute line 05

#### 10. A new summer inter was frustrated with CUDA. He has been complaining that CUDA is very tedious. He had to declare many functions that he plans to execute on both the host and the device twice, once as host function and once as a device function. What is your response?

One can use both qualifiers, `__host__` and `__device__`,  in a function declaration.
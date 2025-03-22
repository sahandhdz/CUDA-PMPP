Chapter 4 - Compute architecture and scheduling

## Exercises

#### 1. Consider the following CUDA kernel and the corresponding host function that calls it:

```c++
__global__
void foo_kernel(int* a, int* b){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadIdx.x < 40 || threadIdx.x >= 104){
        b[i] = a[i] + 1;
    }
    if (i%2 == 0){
        a[i] = b[i]*2;
    }
    for (unsigned int j=0; j<5-(i%3); j++){
        b[i] += j;
    }
}

void foo(int* a_d, int* b_d){
    unsigned int N = 1024;
    foo_kernel<<<(N+128-1)/128, 128>>>(a_d, b_d);
}
```

#### a. What is the number of warps per block?

Each block has 128 threads. Assume the warp size of 32, we should have 4 warps per block.

#### b. What is the number of warps in a grid?

There are 8 blocks in the grid. Hence, `8*4=32` warps in total.

#### c. For the statement in line 05:

##### i. How many warps in the grid are active?

Based on the condition in the if-statement, in each block, **warp 0** (0 - 31), **warps 1** (32 - 63), and **warp 3** (96 - 127) are active. Accordingly, in the whole gird, `8*3=24` warps will be active for this line of code.

#####  ii. How many warps in the grid are divergent?

For this specific line, in each block, warp 1 (threads 32 to 63) shows divergence as threads 32 to 39 will pass the condition at line 04 while the rest of the threads do not. Similarly, warp 3 (threads 96 to 127) also shows divergence. Accordingly, `8*2=16` warps in the grid are divergent.

##### iii. What is the SIMD efficiency (in %) of warp 0 of block 0?

SIMD efficiency for this warp is %100 as all threads (0-31) in this warp are active at this lie of code.

##### iv. What is the SIMD efficiency (in %) of warp 1 of block 0?

SIMD efficiency for this warp is %25 as only the first 8 threads (32 to 39) are active.

##### v. What is the SIMD efficiency (in %) of warp 3 of block 0?

SIMD efficiency for this warp is %75 as only the last 24 threads (104 to 127) are active.


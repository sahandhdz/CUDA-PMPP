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

#### d. For the statement on line 08:

##### i. How many warps in the grid are active?

Every warp in the grid has at least one active thread. So 32 warps (all) in the grid are active.

##### ii. How many warps in the grid are divergent?

All warps are partially active, as half of the threads pass the condition and the rest are inactive. Accordingly, all 32 warps are divergent.

##### iii. What is the SIMD efficiency (in %) of warp 0 of block 0?

SIMD efficiency for this warp, at this line of code, is %50 as only half of the threads (0, 2, 4, ..., 30) process this statement.

#### e. For the loop on line 10:

##### i. How many iterations have no divergence?

A warps is non-divergent if all 32 threads have the same `i%3` value. This only happens if all `i` values in the warp are spaced by 3. Since I incrementally increases within a warp, this condition never met for any warp. So the right answer seems to be all warps. However, I have a little doubt that the question is actually asking for the number of iterations happening in each thread inside a warp. In this case, it is obvious that all threads iterate over `j=0,1,2`, there should be no divergence in the first 3 iterations.

##### ii. How many iterations have divergence?

for iterations `j=3, 4` we observe divergence. But, as explained above there is no agreement in the loop condition between the threads inside the warp and the warp is divergent.

#### 2. For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

Four blocks are required to cover the whole vector of 2000 elements. So, we will have `4*512=2048` threads in  total. It should be noted that the last 48 threads in the last block are idle.

#### 3. For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?

Only the warp that covers the element 1985 to 1999 of the vector have divergence. So the answer is 1 warp.

#### 4. Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, .6, and 2.9; they spend the rest of their time waiting for the barrier. What percentage of the threads total execution time is spent waiting for the barrier?

The longest time a thread takes to execute its task is 3.0 seconds. All other threads reach the barrier before this and start their waiting time. The to total execution time is `3*8 = 24s` and the total working time is `2+2.3+3+2.8+2.4+1.9+0.6+2.9=17.9s`. Hence, the waiting time is `24-17.9=6.1s`. So about %25.4 of the execution time goes for waiting.

#### 5. A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the `__syncthreads()` instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain.

It is not a good idea. The programmer may think that 32 threads form a warp and they just perform all instructions at the same time, so there will be no need for synchronization. However, there might be some kind divergence in the instructions, which necessitates a synchronization at some point to guarantee convergence. Moreover, the shared memory read/write action  also requires synchronization. There might be the some cases that the threads in a warp, collaboratively, read data from global memory and write it to the shared memory for later calculations. In this case, before the calculation starts, we need to make sure that all required data have been moved to the shared memory, so `__syncthreads()` will be necessary.

#### 6. If a CUDA device's SM can take up to 1536 threads and up to 4 threads blocks, which of the following block configurations would result in the most number of threads per block?

#### a) 128 threads per block. b) 256 threads per block. c) 512 threads per block. d) 1024 threads per block.

512 thread per block seems to be the best configuration. The reason is that in this case we can choose 3 blocks, and as a result, we will be able to use `3*512=1536` threads, which is the total number of available threads in a SM. 

#### 7. Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level.

#### a. 8 blocks with 128 threads each.

Assuming the warp size of 32 threads, the SM in total have 64 warps. In the assignment above, we have 8 blocks and 4 warps per block, resulting in a total of 32 warps. Hence, the occupancy is `32/64 = 05`.

#### b. 16 blocks with 64 threads each.

16 blocks with 2 warps per block, results in 32 warps in total. the occupancy level is again `32/64=0.5`.

#### c. 32 blocks with 32 threads each.

So, 32 warps in total. Occupancy will be `32/64=0.5`.

#### d. 64 blocks with 32 threads each.

This will result in 64 warps in total. The occupancy is full, `64/64=1`.

#### e. 32 blocks with 64 threads each.

We have 2 warps per block, resulting in a total of `32*2=64` warps. The occupancy is full.

#### 8. Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64k (65,536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.

#### a. The kernel used 128 threads per block and 30 registers per threads.

Yes, the kernel can achieve full occupancy. With 16 blocks, each with 128 threads, there will be a total of 2048 threads per SM. 30 registers per threads then result in 61,440 threads which is less than the limit of 65,5346 threads. So, the kernel can use all threads in SM, showing full occupancy.

#### b. The kernel uses 32 threads per block and 29 registers per thread.

#### c. The kernel uses 256 threads per block and 34 registers per thread.






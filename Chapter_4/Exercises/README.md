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






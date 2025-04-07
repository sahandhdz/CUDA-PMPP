# Chapter 5 - Memory architecture and data locality

## Exercises

#### 1. Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.

Using shared memory in this case does not help, and may even result in a slower performance. In the element-wise addition of two given matrices A and B, resulting in C, if we assume that each thread calculate one element of the matrix C, two different threads use different elements from A and B, so there are no shared usage between the threads and as a result no chance for cooperation, in contrast to what we saw in the case of matrix multiplication.

#### 2. Draw the equivalent of Fig. 5.7 for a 8 x 8 matrix multiplication with 2 x 2 tiling and 4 x 4 tiling. Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.

You can think like this. For the case of 8 x 8 matrices, with a tiling of size 1 (what we saw in chapter 2 for example), each thread have to read 8 elements of a row of matrix A and 8 elements of a column of matrix B. So, in total 16 readings from global memory is required.

Now let increase tiling size to 2. In this case, the 2 x 2 block (4 threads) should compute 4 elements of the output matrix. To do so, 2 rows from A and 2 columns from be have to be read from global memory (in a collaborative way). So, in total there will be 32 reads from global memory. Accordingly, there will be `32/4=8` reads per threads. Compared to the case of 1 x 1 tile, we see %50 less reads.

Now moving to the case of 4 x 4 tiling, we can see that 64 readings from global memory is needed (4 rows of matrix A and 4 columns of matrix B). So, there will be `64/16=4` readings per thread (16 is the number of total treads in the block/tile). It confirms that the number of total accesses to global memory has been divided by 4.

#### 3. What type of incorrect execution behavior can happen if one forgot to use one or both `__syncthreads()` in the kernel of Fig 5.9?

The two `_syncthreads()` commands used in this kernel are used to make sure that the collaborative work of reading from global memory and writing into the shared memory are being done correctly. If we omit the  the first thread synchronization action, the process of reading all required data from global memory may not complete on time, so the threads proceed to calculate the `Pvalue` without the required data which obviously can lead to incorrect calculations. 

The second synchronization action, actually makes sure that all the required calculations with the available data on the shared memory are done and now the threads can move on to the next step, which is reading another tile of float numbers from global memory **or** writing the final value into the the output matrix. If we omit this synchronization, some threads may start transferring data from global memory to the shared memory while some other threads are still working on the previous data. This will eventuate in erroneous calculations.


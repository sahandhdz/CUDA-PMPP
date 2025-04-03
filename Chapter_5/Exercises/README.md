# Chapter 5 - Memory architecture and data locality

## Exercises

#### 1. Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.

Using shared memory in this case does not help, and may even result in a slower performance. In the element-wise addition of two given matrices A and B, resulting in C, if we assume that each thread calculate one element of the matrix C, two different threads use different elements from A and B, so there are no shared usage between the threads and as a result no chance for cooperation, in contrast to what we saw in the case of matrix multiplication.
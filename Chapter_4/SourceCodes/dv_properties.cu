#include <stdio.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

int main(){
    // Counting the number of existing devices (GPUs)! I have one, but, anyway ...
    int devCount;
    cudaGetDeviceCount(&devCount);
    std::cout << "The number of devices: " << devCount << std::endl;

    cudaDeviceProp devProp;
    for (unsigned int i = 0; i<devCount; i++){
        cudaGetDeviceProperties(&devProp, i);
    }
    
    std::cout << "Number of SMs: " << devProp.multiProcessorCount << std::endl;
    std::cout << "Number of registers available in each SM: " << devProp.regsPerBlock << std::endl; 
    std::cout << "Maximum number of threads allowed in a block: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "Warp size: " << devProp.warpSize << std::endl;
    std::cout << "-----------\n";
    std::cout << "Maximum number of threads allowed along x-direction of a block: " << devProp.maxThreadsDim[0] << std::endl;
    std::cout << "Maximum number of threads allowed along y-direction of a block: " << devProp.maxThreadsDim[1] << std::endl;
    std::cout << "Maximum number of threads allowed along z-direction of a block: " << devProp.maxThreadsDim[2] << std::endl;
    std::cout << "-----------\n";
    std::cout << "Maximum number of blocks allowed along x-direction og a grid: " << devProp.maxGridSize[0] << std::endl;
    std::cout << "Maximum number of blocks allowed along y-direction og a grid: " << devProp.maxGridSize[1] << std::endl;
    std::cout << "Maximum number of blocks allowed along z-direction og a grid: " << devProp.maxGridSize[2] << std::endl;

    /*
    The number of devices: 1
    Number of SMs: 16
    Number of registers available in each SM: 65536
    Maximum number of threads allowed in a block: 1024
    Warp size: 32
    -----------
    Maximum number of threads allowed along x-direction of a block: 1024
    Maximum number of threads allowed along y-direction of a block: 1024
    Maximum number of threads allowed along z-direction of a block: 64
    -----------
    Maximum number of blocks allowed along x-direction og a grid: 2147483647
    Maximum number of blocks allowed along y-direction og a grid: 65535
    Maximum number of blocks allowed along z-direction og a grid: 65535
    */
 
    return 0;
}
#include <stdio.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION 
#include "stb_image_write.h"
#include <iostream>


__global__
void colorToGrayscaleKernel(unsigned char* img_color, unsigned char* img_gray, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col <width && row < height){
        int gray_idx = row*width + col;

        int rgb_idx = gray_idx * 3;

        unsigned char r = img_color[rgb_idx];
        unsigned char g = img_color[rgb_idx + 1];
        unsigned char b = img_color[rgb_idx + 2];

        img_gray[gray_idx] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

void colorToGrayscale(unsigned char* image_color_h, unsigned char* image_gray_h, int width, int height){
    int size_color = 3 * width * height * sizeof(unsigned char);
    int size_gray = width * height * sizeof(unsigned char);

    unsigned char* image_color_d;
    unsigned char* image_gray_d;

    cudaMalloc((void**)&image_color_d, size_color);
    cudaMalloc((void**)&image_gray_d, size_gray);

    cudaMemcpy(image_color_d, image_color_h, size_color, cudaMemcpyHostToDevice);

    dim3 dimgrid(ceil(width/32.0), ceil(height/32.0), 1);
    dim3 dimblock(32, 32, 1);

    colorToGrayscaleKernel<<<dimgrid, dimblock>>>(image_color_d, image_gray_d, width, height);

    cudaMemcpy(image_gray_h, image_gray_d, size_gray, cudaMemcpyDeviceToHost);

    cudaFree(image_color_d);
    cudaFree(image_gray_d);
}



void save_grayscale_image(const std::string& filename, const unsigned char* img_data, int width, int height) {
    if (stbi_write_jpg(filename.c_str(), width, height, 1, img_data, 100)) {
        std::cout << "Image saved: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save image!" << std::endl;
    }
}


int main(){
    int width, height, channels;
    unsigned char* img = stbi_load("ladybug.jpg", &width, &height, &channels, 0);
    
    if (!img) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    unsigned char* img_gray = new unsigned char[width*height];

    colorToGrayscale(img, img_gray, width, height);

    std::string filename = "ladybug_Grey.jpg";

    save_grayscale_image(filename, img_gray, width, height);
    
    
    // Free image memory
    stbi_image_free(img);
    delete[] img_gray;
    return 0;
}
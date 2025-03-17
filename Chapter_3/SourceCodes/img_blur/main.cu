#include <stdio.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Including the impelemntation of stb_image.h for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
// Including the implementation of stb_image_write.h for image saving
#define STB_IMAGE_WRITE_IMPLEMENTATION 
#include "stb_image_write.h"

#include <iostream>


__global__
void blurKernel(unsigned char* img_input, unsigned char* img_output, int width, int height, int blur_size){
    // col and row indices, specifying the unqiue coordinate assigned to each thread (we use a 2D thread mapping)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height){
        int r_sum = 0;
        int g_sum = 0;
        int b_sum = 0;
        int pix_num = 0;

        for (int i=-blur_size; i<blur_size+1; i++){
            for (int j=-blur_size; j<blur_size+1; j++){
                int current_row = row+i;
                int current_col = col+j;

                if (current_row >=0 && current_row <height && current_col >=0 && current_col < width){
                    r_sum += img_input[current_row*width*3 + current_col*3];
                    g_sum += img_input[current_row*width*3 + current_col*3 + 1];
                    b_sum += img_input[current_row*width*3 + current_col*3 + 2];
                    pix_num ++;
                }
            }
        }
        img_output[row*width*3 + col] = (unsigned char)(r_sum/pix_num);
        img_output[row*width*3 + col + 1] = (unsigned char)(g_sum/pix_num);
        img_output[row*width*3 + col + 2] = (unsigned char)(b_sum/pix_num);
    }
}


int main(){
    // Reading the image using stb_image.h header by Sean T. Barrett (that's why it starts with stb!).
    // (github.com/nothings/stb)
    int width, height, channels;
    unsigned char* img = stbi_load("ladybug.jpg", &width, &height, &channels, 0);
    
    if (!img) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }



    return 0;
}
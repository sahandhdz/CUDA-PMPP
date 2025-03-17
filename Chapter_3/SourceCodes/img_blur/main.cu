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
        img_output[row*width*3 + col*3] = (unsigned char)(r_sum/pix_num);
        img_output[row*width*3 + col*3 + 1] = (unsigned char)(g_sum/pix_num);
        img_output[row*width*3 + col*3 + 2] = (unsigned char)(b_sum/pix_num);
    }
}

void blurImage(unsigned char* image_input_h, unsigned char* image_output_h, int width, int height, int blur_size){
    int image_size = 3 * width * height * sizeof(unsigned char);;

    unsigned char* image_input_d;
    unsigned char* image_output_d;

    cudaMalloc((void**)&image_input_d, image_size);
    cudaMalloc((void**)&image_output_d, image_size);

    cudaMemcpy(image_input_d, image_input_h, image_size, cudaMemcpyHostToDevice);

    dim3 dimgrid(ceil(width/32.0), ceil(height/32.0), 1);
    dim3 dimblock(32, 32, 1);

    blurKernel<<<dimgrid, dimblock>>>(image_input_d, image_output_d, width, height, blur_size);

    cudaMemcpy(image_output_h, image_output_d, image_size, cudaMemcpyDeviceToHost);

    cudaFree(image_input_d);
    cudaFree(image_input_h);
}

// **Save RGB Image as JPG**
void save_rgb_image(const std::string& filename, const unsigned char* img_data, int width, int height) {
    if (stbi_write_jpg(filename.c_str(), width, height, 3, img_data, 100)) { // Change 1 → 3 for RGB
        std::cout << "Image saved: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save image!" << std::endl;
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

    unsigned char* img_blur = new unsigned char[width*height*3];

    int blur_size = 8;
    blurImage(img, img_blur, width, height, blur_size);

    // Saving the final image.
    std::string filename = "ladybug_Grey_blur.jpg";
    save_rgb_image(filename, img_blur, width, height);

    // Free image memory
    stbi_image_free(img);
    delete[] img_blur;

    return 0;
}
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "cuda_utils.cuh"
#include <cstdio>

__global__ void rgb_to_grayscale(
    const unsigned char* __restrict__ rgb,
    unsigned char* __restrict__ gray,
    int width, int height)
{
    // which pixel
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return; // grid is slightly bigger than image, ignore extras

    int gray_idx = row * width + col;
    int rgb_idx  = gray_idx * 3; // 3 bytes per pixel in the input

    float r = rgb[rgb_idx + 0];
    float g = rgb[rgb_idx + 1];
    float b = rgb[rgb_idx + 2];

    gray[gray_idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    const char* input_path  = argv[1];
    const char* output_path = argv[2];

    // load on CPU
    int width, height, channels_in_file;
    unsigned char* h_rgb  = stbi_load(input_path, &width, &height, &channels_in_file, 3);
    if (!h_rgb) {
        fprintf(stderr, "Failed to load: %s\n", input_path);
        return 1;
    }
    unsigned char* h_gray = (unsigned char*)malloc(width * height);

    // allocate on GPU and send input over
    unsigned char* d_rgb;
    unsigned char* d_gray;
    CUDA_CHECK(cudaMalloc(&d_rgb,  width * height * 3));
    CUDA_CHECK(cudaMalloc(&d_gray, width * height));
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, width * height * 3, cudaMemcpyHostToDevice));

    // 16x16 = 256 threads per block, ceiling division so we cover the whole image
    dim3 block(16, 16);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    rgb_to_grayscale<<<grid, block>>>(d_rgb, d_gray, width, height);
    CUDA_CHECK(cudaDeviceSynchronize()); // wait for GPU, also catches kernel errors

    // get result back and save
    CUDA_CHECK(cudaMemcpy(h_gray, d_gray, width * height, cudaMemcpyDeviceToHost));
    stbi_write_png(output_path, width, height, 1, h_gray, width);
    printf("Saved: %s\n", output_path);

    // free everything, C++ won't do it for you
    cudaFree(d_rgb);
    cudaFree(d_gray);
    stbi_image_free(h_rgb);
    free(h_gray);

    return 0;
}
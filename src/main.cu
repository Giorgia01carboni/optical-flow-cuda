#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "cuda_utils.cuh"
#include "horn_schunck.cuh"
#include "pyramid.cuh"
#include "warp.cuh"
#include "blend.cuh"
#include <cstdio>
#include <cmath>   // sqrtf

__global__ void rgb_to_grayscale(
    const unsigned char* __restrict__ rgb,
    unsigned char* __restrict__ gray,
    int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int gray_idx = row * width + col;
    int rgb_idx  = gray_idx * 3;

    float r = rgb[rgb_idx + 0];
    float g = rgb[rgb_idx + 1];
    float b = rgb[rgb_idx + 2];
    gray[gray_idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

// visualise flow as a grayscale magnitude image so we can actually see it
__global__ void flow_to_image(
    const float* __restrict__ u, // horizontal speed
    const float* __restrict__ v, // vertical speed
    unsigned char* __restrict__ out,
    int width, int height, // frame dimensions
    float scale) // multiply magnitude by this to bring it into 0-255 range
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate 
    int row = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate
    if (col >= width || row >= height) return; // out of bounds check

    int idx = row * width + col; // Index for this pixel
    float mag = sqrtf(u[idx]*u[idx] + v[idx]*v[idx]); // magnitude of flow vector: sqrt(u² + v²). How fast is the motion regardless of direction
    float val = fminf(mag * scale, 255.0f); // clamp to [0, 255]
    out[idx] = (unsigned char)val;
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        printf("Usage: %s <frame1> <frame2> <output_flow>\n", argv[0]);
        return 1;
    }

    const char* path1   = argv[1];
    const char* path2   = argv[2];
    const char* out_path = argv[3];

    // load both frames on CPU
    int w1, h1, c1, w2, h2, c2;
    unsigned char* h_rgb1 = stbi_load(path1, &w1, &h1, &c1, 3);
    unsigned char* h_rgb2 = stbi_load(path2, &w2, &h2, &c2, 3);

    if (!h_rgb1 || !h_rgb2) {
        fprintf(stderr, "Failed to load images\n");
        return 1;
    }
    if (w1 != w2 || h1 != h2) {
        fprintf(stderr, "Frames must be the same size\n");
        return 1;
    }

    int width = w1, height = h1;
    int n_rgb  = width * height * 3;
    int n_gray = width * height;
    printf("Frames: %dx%d\n", width, height);

    // allocate and send both RGB frames to GPU
    unsigned char *d_rgb1, *d_rgb2;
    CUDA_CHECK(cudaMalloc(&d_rgb1, n_rgb));
    CUDA_CHECK(cudaMalloc(&d_rgb2, n_rgb));
    CUDA_CHECK(cudaMemcpy(d_rgb1, h_rgb1, n_rgb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rgb2, h_rgb2, n_rgb, cudaMemcpyHostToDevice));

    // grayscale buffers on GPU
    unsigned char *d_gray1, *d_gray2;
    CUDA_CHECK(cudaMalloc(&d_gray1, n_gray));
    CUDA_CHECK(cudaMalloc(&d_gray2, n_gray));

    dim3 block(16, 16);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    rgb_to_grayscale<<<grid, block>>>(d_rgb1, d_gray1, width, height);
    rgb_to_grayscale<<<grid, block>>>(d_rgb2, d_gray2, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // optical flow between the two frames
    // flow buffers —> one float per pixel per direction
    float *d_u, *d_v;
    CUDA_CHECK(cudaMalloc(&d_u, n_gray * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, n_gray * sizeof(float)));

    // run Horn-Schunck: alpha=10 (smoothness), 100 iterations REMEMBER TO CHECK THIS AGAIN
    pyramidal_horn_schunck(d_gray1, d_gray2, d_u, d_v, width, height, 10.0f, 10, 4);
    printf("Pyramidal Horn-Schunck computed\n");

    // warp frame1 forward by t=0.5 and frame2 backward by 1-t=0.5
    unsigned char *d_warped1, *d_warped2;
    CUDA_CHECK(cudaMalloc(&d_warped1, n_rgb));
    CUDA_CHECK(cudaMalloc(&d_warped2, n_rgb));

    warp_frame(d_rgb1, d_u, d_v, d_warped1, width, height,  0.5f);
    warp_frame(d_rgb2, d_u, d_v, d_warped2, width, height, -0.5f); // negative = backward

    // blend the two warped frames into the final interpolated frame
    unsigned char* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, n_rgb));
    blend_frames(d_warped1, d_warped2, d_out, width, height);
    printf("Interpolated frame ready\n");


    unsigned char* h_out = (unsigned char*)malloc(n_rgb);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, n_rgb, cudaMemcpyDeviceToHost));

    stbi_write_png(out_path, width, height, 3, h_out, width * 3);

    printf("Saved: %s\n", out_path);

    // free everything
    cudaFree(d_rgb1); cudaFree(d_rgb2);
    cudaFree(d_gray1); cudaFree(d_gray2);
    cudaFree(d_u); cudaFree(d_v);
    cudaFree(d_warped1); cudaFree(d_warped2);
    cudaFree(d_out);
    stbi_image_free(h_rgb1);
    stbi_image_free(h_rgb2);
    free(h_out);

    return 0;
}
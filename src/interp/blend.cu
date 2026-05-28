#include "blend.cuh"
#include "cuda_utils.cuh"

// simple 50/50 average of two RGB frames
__global__ void blend_kernel(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ b,
    unsigned char* __restrict__ out,
    int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int base = (row * width + col) * 3; // thread index for the first color channel of this pixel
    for (int c = 0; c < 3; c++) {
        // Average 2 frames per pixel and per color channel
        out[base + c] = (unsigned char)((a[base + c] + b[base + c]) * 0.5f);
    }
}

void blend_frames(
    const unsigned char* d_frame_a,
    const unsigned char* d_frame_b,
    unsigned char* d_out,
    int width, int height)
{
    dim3 block(16, 16);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    blend_kernel<<<grid, block>>>(d_frame_a, d_frame_b, d_out, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
}
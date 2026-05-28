#include "warp.cuh"
#include "cuda_utils.cuh"

// one thread per output pixel
// d_u, d_v motion vector for every pixel.
// Using backwards warping: for each output pixel check the corresponding location in source frame by going backwards along the flow vector
// then samples that location using bilinear interpolation
// Bidirectional warp: t=0 means all pixels come from the source frame, t=1 means all pixels come from the destination frame, t=0.5 means halfway between the two frames.
// warp from frame1 forward and from frame2 backward, then blend the two results
__global__ void warp_kernel(
    const unsigned char* __restrict__ src,
    const float* __restrict__ d_u,
    const float* __restrict__ d_v,
    unsigned char* __restrict__ out,
    int width, int height,
    float t)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int idx = row * width + col;

    // Find where this pixel came from by going backwards along the flow
    float src_x = col - d_u[idx] * t;
    float src_y = row - d_v[idx] * t;

    // clamp to image borders (to not go out of bounds)
    src_x = fmaxf(0.0f, fminf(src_x, width  - 1.001f));
    src_y = fmaxf(0.0f, fminf(src_y, height - 1.001f));

    // bilinear interpolation 
    // blend the 4 surrounding pixels 
    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, width  - 1);
    int y1 = min(y0 + 1, height - 1);

    float fx = src_x - x0;  // fractional part: how far between x0 and x1
    float fy = src_y - y0;

    // weights for the 4 corners
    float w00 = (1-fx) * (1-fy);
    float w10 =    fx  * (1-fy);
    float w01 = (1-fx) *    fy;
    float w11 =    fx  *    fy;

    // sample all 3 channels (R, G, B) for each of the 4 corners and blend
    for (int c = 0; c < 3; c++) {
        float val = w00 * src[( y0*width + x0)*3 + c]
                  + w10 * src[( y0*width + x1)*3 + c]
                  + w01 * src[( y1*width + x0)*3 + c]
                  + w11 * src[( y1*width + x1)*3 + c];

        out[idx*3 + c] = (unsigned char)fminf(val, 255.0f);
    }
}

void warp_frame(
    const unsigned char* d_src,
    const float* d_u,
    const float* d_v,
    unsigned char* d_out,
    int width, int height,
    float t)
{
    dim3 block(16, 16);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    warp_kernel<<<grid, block>>>(d_src, d_u, d_v, d_out, width, height, t);
    CUDA_CHECK(cudaDeviceSynchronize());
}
#include "pyramid.cuh"
#include "horn_schunck.cuh"
#include "cuda_utils.cuh"
#include <cstdlib>
#include <vector>

// The kernel is separable (horizontal and vertical passes) 3x3
// blurs before downsampling to avoid aliasing (high frequency noise folding in)

__global__ void gaussian_blur(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width, int height) 
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        if (col >= width || row >= height) return;

        // clamp to image borders
        int l = max(col-1, 0), r = min(col+1, width-1);
        int u = max(row-1, 0), d = min(row+1, height-1);

        //Gaussian kernel's weights ( 1 2 1 / 2 4 2 / 1 2 1) / 16
        float val =
            1*src[u*width+l] + 2*src[u*width+col] + 1*src[u*width+r] +
            2*src[row*width+l] + 4*src[row*width+col] + 2*src[row*width+r] +
            1*src[d*width+l] + 2*src[d*width+col] + 1*src[d*width+r];

        dst[row*width+col] = (unsigned char)(val / 16.0f);
    }

// Downsample kernel (runs at output resolution, so 1 thread per output pixel)
__global__ void downsample(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int src_width,
    int dst_width, int dst_height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= dst_width || row >= dst_height) return;

    dst[row * dst_width + col] = src[(row*2) * src_width + (col*2)]; // take pixel at 2x the source coordinates
}

// Upsample kernel (runs at output larger res)
// each output pixel copies the nearest flow vector from the smaller res lvel and multiplies by 2 (motion vector scales with resolution)

__global__ void upsample_flow(
    const float* __restrict__ src_u,
    const float* __restrict__ src_v,
    float* __restrict__ dst_u,
    float* __restrict__ dst_v,
    int src_width, int src_height,
    int dst_width, int dst_height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= dst_width || row >= dst_height) return;

    // corresponding position in the smaller flow field
    int src_col = min(col / 2, src_width - 1);
    int src_row = min(row / 2, src_height - 1);

    int src_idx = src_row * src_width + src_col;
    int dst_idx = row * dst_width + col;

    // multiply by 2 to scale the flow vector for the higher resolution (example: 1px motion at half resolution = 2px motion at full resolution)
    dst_u[dst_idx] = src_u[src_idx] * 2.0f;
    dst_v[dst_idx] = src_v[src_idx] * 2.0f;
}

// Put all pyramid levels together: blur + downsample repeatedly

std::vector<PyramidLevel> build_pyramid(
    const unsigned char* d_gray1,
    const unsigned char* d_gray2,
    int width, int height,
    int num_levels)
{
    std::vector<PyramidLevel> pyramid(num_levels);
    // 0 -> full res, num_levels-1 -> smallest res
    pyramid[0] = { (unsigned char*)d_gray1, (unsigned char*)d_gray2, width, height };

    for (int i = 1; i < num_levels; i++) {
        int prev_w = pyramid[i-1].width;
        int prev_h = pyramid[i-1].height;
        int cur_w  = prev_w / 2;
        int cur_h  = prev_h / 2;

        unsigned char *blurred1, *blurred2;
        unsigned char *small1,   *small2;
        CUDA_CHECK(cudaMalloc(&blurred1, prev_w * prev_h));
        CUDA_CHECK(cudaMalloc(&blurred2, prev_w * prev_h));
        CUDA_CHECK(cudaMalloc(&small1,   cur_w  * cur_h));
        CUDA_CHECK(cudaMalloc(&small2,   cur_w  * cur_h));

        dim3 block(16, 16);

        // blur at previous resolution before downsampling
        dim3 grid_prev((prev_w + 15)/16, (prev_h + 15)/16);
        gaussian_blur<<<grid_prev, block>>>(pyramid[i-1].d_frame1, blurred1, prev_w, prev_h);
        gaussian_blur<<<grid_prev, block>>>(pyramid[i-1].d_frame2, blurred2, prev_w, prev_h);
        CUDA_CHECK(cudaDeviceSynchronize());

        // downsample to current resolution
        dim3 grid_cur((cur_w + 15)/16, (cur_h + 15)/16);
        downsample<<<grid_cur, block>>>(blurred1, small1, prev_w, cur_w, cur_h);
        downsample<<<grid_cur, block>>>(blurred2, small2, prev_w, cur_w, cur_h);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(blurred1);
        cudaFree(blurred2);

        pyramid[i] = { small1, small2, cur_w, cur_h };
    }
    
    return pyramid;
}

void free_pyramid(std::vector<PyramidLevel>& pyramid) {
    // skip level 0 since those point to the original frames 
    for (int i = 1; i < (int)pyramid.size(); i++) {
        cudaFree(pyramid[i].d_frame1);
        cudaFree(pyramid[i].d_frame2);
    }
}

void pyramidal_horn_schunck(
    const unsigned char* d_gray1,
    const unsigned char* d_gray2,
    float* d_u, float* d_v,
    int width, int height,
    float alpha,
    int iterations_per_level,
    int num_levels)
{
    auto pyramid = build_pyramid(d_gray1, d_gray2, width, height, num_levels);

    // start at coarsest level with zero flow
    int cw = pyramid[num_levels-1].width;
    int ch = pyramid[num_levels-1].height;

    float *d_u_coarse, *d_v_coarse;
    CUDA_CHECK(cudaMalloc(&d_u_coarse, cw * ch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_coarse, cw * ch * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_u_coarse, 0, cw * ch * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_coarse, 0, cw * ch * sizeof(float)));

    // run Horn-Schunck at the coarsest level
    horn_schunck(
        pyramid[num_levels-1].d_frame1,
        pyramid[num_levels-1].d_frame2,
        d_u_coarse, d_v_coarse,
        cw, ch, alpha, iterations_per_level
    );

    // refine level by level from coarse to fine
    for (int lvl = num_levels-2; lvl >= 0; lvl--) {
        int fw = pyramid[lvl].width;
        int fh = pyramid[lvl].height;

        // upsample flow from coarser level to this level's resolution
        float *d_u_fine, *d_v_fine;
        CUDA_CHECK(cudaMalloc(&d_u_fine, fw * fh * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_fine, fw * fh * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((fw+15)/16, (fh+15)/16);
        upsample_flow<<<grid, block>>>(d_u_coarse, d_v_coarse, d_u_fine, d_v_fine, cw, ch, fw, fh);
        CUDA_CHECK(cudaDeviceSynchronize());

        // refine with Horn-Schunck at this level, starting from the upsampled flow
        horn_schunck(
            pyramid[lvl].d_frame1,
            pyramid[lvl].d_frame2,
            d_u_fine, d_v_fine,
            fw, fh, alpha, iterations_per_level
        );

        cudaFree(d_u_coarse);
        cudaFree(d_v_coarse);

        d_u_coarse = d_u_fine;
        d_v_coarse = d_v_fine;
        cw = fw;
    }

    // d_u_coarse/d_v_coarse now hold the full resolution result
    int n = width * height * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_u, d_u_coarse, n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, d_v_coarse, n, cudaMemcpyDeviceToDevice));

    cudaFree(d_u_coarse);
    cudaFree(d_v_coarse);
    free_pyramid(pyramid);

}
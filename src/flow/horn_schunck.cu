#include "horn_schunck.cuh"
#include "cuda_utils.cuh"
#include <cstdlib>

// Kernel 1: compute image gradients
// Ix, Iy: spatial gradients (how brightness changes in x and y within a frame)
// It:     temporal gradient (how brightness changes between the two frames)
// All needed for the Jacobi update.

__global__ void compute_gradients(
    const unsigned char* __restrict__ f1,
    const unsigned char* __restrict__ f2,
    float* __restrict__ Ix,
    float* __restrict__ Iy,
    float* __restrict__ It,
    int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    // clamp neighbours to image borders to avoid reading from out of bounds
    int left  = max(col - 1, 0);
    int right = min(col + 1, width - 1);
    int up    = max(row - 1, 0);
    int down  = min(row + 1, height - 1);

    // central differences: estimate how brightness changes in each direction
    // Average over both frames for a more stable estimate
    float f1_r = f1[row  * width + right];
    float f1_l = f1[row  * width + left];
    float f2_r = f2[row  * width + right];
    float f2_l = f2[row  * width + left];

    float f1_d = f1[down * width + col];
    float f1_u = f1[up   * width + col];
    float f2_d = f2[down * width + col];
    float f2_u = f2[up   * width + col];

    int idx = row * width + col;
    Ix[idx] = ((f1_r - f1_l) + (f2_r - f2_l)) * 0.25f;
    Iy[idx] = ((f1_d - f1_u) + (f2_d - f2_u)) * 0.25f;
    It[idx] = (float)f2[idx] - (float)f1[idx]; // temporal: frame2 - frame1
}


// Kernel 2: one Jacobi iteration 
// reads u_in, v_in (previous iteration), writes u_out, v_out (this iteration)
// (we alternate between two buffers to avoid reading values we just wrote)

__global__ void hs_iteration(
    const float* __restrict__ Ix,
    const float* __restrict__ Iy,
    const float* __restrict__ It,
    const float* __restrict__ u_in,
    const float* __restrict__ v_in,
    float* __restrict__ u_out,
    float* __restrict__ v_out,
    int width, int height,
    float alpha_sq) // α^2 precomputed on host
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    int idx = row * width + col;

    // 4-pixel neighbourhood average (clamp at borders)
    int left  = row * width + max(col - 1, 0);
    int right = row * width + min(col + 1, width - 1);
    int up    = max(row - 1, 0) * width + col;
    int down  = min(row + 1, height - 1) * width + col;

    float u_avg = (u_in[left] + u_in[right] + u_in[up] + u_in[down]) * 0.25f;
    float v_avg = (v_in[left] + v_in[right] + v_in[up] + v_in[down]) * 0.25f;

    // the update rule — P is how much we need to correct the flow estimate
    float ix = Ix[idx];
    float iy = Iy[idx];
    float it = It[idx];

    float P = (ix * u_avg + iy * v_avg + it) / (alpha_sq + ix*ix + iy*iy);

    u_out[idx] = u_avg - ix * P;
    v_out[idx] = v_avg - iy * P;
}


// ── Host function: wire everything together ───────────────────────────────────

void horn_schunck(
    const unsigned char* d_frame1,
    const unsigned char* d_frame2,
    float* d_u, // output: horizontal flow
    float* d_v, // output: vertical flow
    int width, int height,
    float alpha, // if smoothness is higher = blurrier but more stable flow
    int iterations)
{
    int n = width * height; // number of pixels
    int bytes_f = n * sizeof(float);

    // allocate gradient buffers on GPU
    float *d_Ix, *d_Iy, *d_It; // Ix = left/right, Iy = up/down, It = frame1 vs frame2
    CUDA_CHECK(cudaMalloc(&d_Ix, bytes_f));
    CUDA_CHECK(cudaMalloc(&d_Iy, bytes_f));
    CUDA_CHECK(cudaMalloc(&d_It, bytes_f));

    // check between the buffers: alternate which one is "current" each iteration
    // to avoids threads reading values that other threads already updated
    float *d_u2, *d_v2;
    CUDA_CHECK(cudaMalloc(&d_u2, bytes_f));
    CUDA_CHECK(cudaMalloc(&d_v2, bytes_f));

    // start with zero flow (no motion initially)
    CUDA_CHECK(cudaMemset(d_u,  0, bytes_f));
    CUDA_CHECK(cudaMemset(d_v,  0, bytes_f));
    CUDA_CHECK(cudaMemset(d_u2, 0, bytes_f));
    CUDA_CHECK(cudaMemset(d_v2, 0, bytes_f));

    // Divide the image into blocks of 16x16=256 thhreads.
    dim3 block(16, 16); 
    // Calculate how many blocks I need to cover the whole image. 
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    // compute gradients once (they don't change between iterations)
    compute_gradients<<<grid, block>>>(d_frame1, d_frame2, d_Ix, d_Iy, d_It, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    float alpha_sq = alpha * alpha;

    // iteration: each pass refines the flow estimate
    for (int i = 0; i < iterations; i++) {
        if (i % 2 == 0) {
            // even: read from (d_u, d_v), write to (d_u2, d_v2)
            hs_iteration<<<grid, block>>>(d_Ix, d_Iy, d_It, d_u, d_v, d_u2, d_v2, width, height, alpha_sq);
        } else {
            // odd: read from (d_u2, d_v2), write to (d_u, d_v)
            hs_iteration<<<grid, block>>>(d_Ix, d_Iy, d_It, d_u2, d_v2, d_u, d_v, width, height, alpha_sq);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // if we did an odd number of iterations the final result is in d_u2 and d_v2,
    // copy it back to the output buffers the caller expects
    if (iterations % 2 != 0) {
        CUDA_CHECK(cudaMemcpy(d_u, d_u2, bytes_f, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, d_v2, bytes_f, cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_Ix);
    cudaFree(d_Iy);
    cudaFree(d_It);
    cudaFree(d_u2);
    cudaFree(d_v2);
}
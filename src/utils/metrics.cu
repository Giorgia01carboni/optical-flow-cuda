#include "metrics.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// 1) Reading the .flo file (runs on CPU)

float* load_groundtruth_flow(const char* path, int* out_width, int* out_height)
{
    FILE* file = fopen(path, "rb");   // "rb" = read, binary (not a text file!)
    if (!file) {
        fprintf(stderr, "Cannot open .flo file: %s\n", path);
        return nullptr;
    }

    // The file starts with a standard number: a known float used as a sanity
    // check. This is needed just to check if the file is valid (e.g. not a text file with the wrong extension)
    float magic_number;
    fread(&magic_number, sizeof(float), 1, file);
    if (magic_number != 202021.25f) {
        fprintf(stderr, "Not a valid .flo file (number mismatch)\n");
        fclose(file);
        return nullptr;
    }

    // Write the width and height into the output parameters (caller passed their addresses).
    fread(out_width,  sizeof(int), 1, file);
    fread(out_height, sizeof(int), 1, file);

    int pixel_count = (*out_width) * (*out_height);

    // Two floats per pixel (u and v), stored interleaved (check metrics.cuh)
    int float_count = pixel_count * 2;
    float* flow_data = (float*)malloc(float_count * sizeof(float));

    
    fread(flow_data, sizeof(float), float_count, file);

    fclose(file);
    return flow_data;   
}


// 2) Per-pixel endpoint error (runs on GPU, one thread per pixel)

__global__ void endpoint_error_kernel(
    const float* __restrict__ u,            // our flow, horizontal
    const float* __restrict__ v,            // our flow, vertical
    const float* __restrict__ groundtruth,  // true flow, interleaved 
    float* __restrict__ endpoint_errors,    // OUT: error value for each pixel
    float* __restrict__ valid_mask,         // OUT: 1 = count this pixel, 0 = skip
    int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    int idx = row * width + col;

    // Interleaved layout: pixel idx -> u at (2*idx), v at (2*idx + 1).
    float true_u = groundtruth[2 * idx];
    float true_v = groundtruth[2 * idx + 1];

    // Middlebury marks pixels with no reliable ground truth using huge values
    // (around 1e9). Skip these pixels in the evaluation by writing 0 error and 0 valid_mask.
    if (fabsf(true_u) > 1e9f || fabsf(true_v) > 1e9f) {
        endpoint_errors[idx] = 0.0f;
        valid_mask[idx]      = 0.0f;
        return;
    }

    // Endpoint error = Euclidean distance between the two motion vectors.
    float diff_u = u[idx] - true_u;
    float diff_v = v[idx] - true_v;
    endpoint_errors[idx] = sqrtf(diff_u * diff_u + diff_v * diff_v);
    valid_mask[idx]      = 1.0f;
}


// 3) PARALLEL REDUCTION (GPU): sum the per-pixel errors AND count valid pixels.
//    Each block sums its slice in shared memory by halving the active
//    threads every step, then writes one partial result. Done in 2 passes.

#define REDUCE_THREADS 256

__global__ void reduce_sum_kernel(
    const float* __restrict__ errors,    // values to sum
    const float* __restrict__ valid,     // 1/0 mask to sum (counts valid pixels)
    float* __restrict__ partial_errors,  // OUT: one sum per block
    float* __restrict__ partial_counts,  // OUT: one count per block
    int n)
{
    __shared__ float s_err[REDUCE_THREADS];   // scratch: errors for this block
    __shared__ float s_cnt[REDUCE_THREADS];   // scratch: counts for this block

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // load my element into shared memory (0 if past the end)
    s_err[tid] = (i < n) ? errors[i] : 0.0f;
    s_cnt[tid] = (i < n) ? valid[i]  : 0.0f;
    __syncthreads();

    // halve the active threads each step: tid 0 ends up with the block total
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_err[tid] += s_err[tid + stride];
            s_cnt[tid] += s_cnt[tid + stride];
        }
        __syncthreads();   // all adds of this step must finish before the next
    }

    // thread 0 writes this block's partial result
    if (tid == 0) {
        partial_errors[blockIdx.x] = s_err[0];
        partial_counts[blockIdx.x] = s_cnt[0];
    }
}


// 4) Putting it together: read GT, run the kernel, reduce on GPU, average

double evaluate_average_endpoint_error(const float* d_u, const float* d_v,
                                       const char* gt_path,
                                       int width, int height)
{
    // read the ground truth into host memory
    int gt_width, gt_height;
    float* h_groundtruth = load_groundtruth_flow(gt_path, &gt_width, &gt_height);
    if (!h_groundtruth) return -1.0;

    // The ground truth must describe the same image size as our flow.
    if (gt_width != width || gt_height != height) {
        fprintf(stderr, "EPE skipped: size mismatch (flow %dx%d vs gt %dx%d)\n",
                width, height, gt_width, gt_height);
        free(h_groundtruth);
        return -1.0;
    }

    int pixel_count = width * height;
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Move ground truth to GPU and allocate the output buffers 
    float *d_groundtruth, *d_endpoint_errors, *d_valid_mask;
    CUDA_CHECK(cudaMalloc(&d_groundtruth,     pixel_count * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_endpoint_errors, pixel_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid_mask,      pixel_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_groundtruth, h_groundtruth,
                          pixel_count * 2 * sizeof(float),
                          cudaMemcpyHostToDevice));

    // compute the per-pixel error on the GPU
    endpoint_error_kernel<<<grid, block>>>(
        d_u, d_v, d_groundtruth,
        d_endpoint_errors, d_valid_mask, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // PARALLEL REDUCTION on GPU 
    // Pass 1: each block reduces its slice into one partial value.
    int num_blocks = (pixel_count + REDUCE_THREADS - 1) / REDUCE_THREADS;

    float *d_partial_errors, *d_partial_counts;
    CUDA_CHECK(cudaMalloc(&d_partial_errors, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_counts, num_blocks * sizeof(float)));

    reduce_sum_kernel<<<num_blocks, REDUCE_THREADS>>>(
        d_endpoint_errors, d_valid_mask,
        d_partial_errors, d_partial_counts, pixel_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pass 2: the partials are few (num_blocks), so just bring them back and
    // finish the sum on CPU. (Cheaper than launching another kernel for so few values so I'll just keep that)
    float* h_partial_errors = (float*)malloc(num_blocks * sizeof(float));
    float* h_partial_counts = (float*)malloc(num_blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial_errors, d_partial_errors,
                          num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_partial_counts, d_partial_counts,
                          num_blocks * sizeof(float), cudaMemcpyDeviceToHost));

    double error_sum = 0.0;
    double valid_pixels = 0.0;
    for (int b = 0; b < num_blocks; b++) {
        error_sum    += h_partial_errors[b];
        valid_pixels += h_partial_counts[b];
    }
    double average_epe = (valid_pixels > 0.0) ? error_sum / valid_pixels : 0.0;

    printf("Average EPE: %.4f px (over %.0f valid pixels)\n",
           average_epe, valid_pixels);

    free(h_partial_errors);
    free(h_partial_counts);
    free(h_groundtruth);
    cudaFree(d_groundtruth);
    cudaFree(d_endpoint_errors);
    cudaFree(d_valid_mask);
    cudaFree(d_partial_errors);
    cudaFree(d_partial_counts);

    return average_epe;
}

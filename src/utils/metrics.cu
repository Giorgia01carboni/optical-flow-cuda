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
    const float* __restrict__ groundtruth,  // true flow, interleaved [u,v,u,v,...]
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


// 3) Putting it together: read GT, run the kernel, average the result

double evaluate_average_endpoint_error(const float* d_u, const float* d_v,
                                       const char* gt_path,
                                       int width, int height)
{
    // --- read the ground truth into host memory ---
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

    // --- compute the per-pixel error on the GPU ---
    endpoint_error_kernel<<<grid, block>>>(
        d_u, d_v, d_groundtruth,
        d_endpoint_errors, d_valid_mask, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- bring the per-pixel results back and average them ---
    // REMEMBER: this final sum is done on the CPU for now. It is a simple reduction
    // and could later become a parallel GPU reduction 
    float* h_endpoint_errors = (float*)malloc(pixel_count * sizeof(float));
    float* h_valid_mask      = (float*)malloc(pixel_count * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_endpoint_errors, d_endpoint_errors,
                          pixel_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_valid_mask, d_valid_mask,
                          pixel_count * sizeof(float), cudaMemcpyDeviceToHost));

    double error_sum = 0.0;
    long   valid_pixels = 0;
    for (int i = 0; i < pixel_count; i++) {
        if (h_valid_mask[i] > 0.5f) {       // only average the valid pixels
            error_sum += h_endpoint_errors[i];
            valid_pixels++;
        }
    }
    double average_epe = (valid_pixels > 0) ? error_sum / valid_pixels : 0.0;

    printf("Average EPE: %.4f px (over %ld valid pixels)\n",
           average_epe, valid_pixels);

    free(h_endpoint_errors);
    free(h_valid_mask);
    free(h_groundtruth);
    cudaFree(d_groundtruth);
    cudaFree(d_endpoint_errors);
    cudaFree(d_valid_mask);

    return average_epe;
}

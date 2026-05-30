#include "cpu_version.cuh"
#include "horn_schunck.cuh"   
#include "cuda_utils.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>   



const int CPU_RUNS = 5;
const int GPU_RUNS = 100;

// clamp an index into [lo, hi]
static inline int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}


// Sequential Horn-Schunck on CPU (just for benchmarking)
void horn_schunck_cpu(
    const unsigned char* frame1, const unsigned char* frame2,
    float* u, float* v,
    int width, int height,
    float alpha, int iterations)
{
    int n = width * height;
    const float inv = 1.0f / 255.0f;     // same [0,1] normalization as the GPU
    float alpha_sq = alpha * alpha;

    // gradients
    std::vector<float> Ix(n), Iy(n), It(n);
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int left  = clampi(col - 1, 0, width  - 1);
            int right = clampi(col + 1, 0, width  - 1);
            int up    = clampi(row - 1, 0, height - 1);
            int down  = clampi(row + 1, 0, height - 1);
            int idx = row * width + col;

            float f1r = frame1[row*width+right], f1l = frame1[row*width+left];
            float f2r = frame2[row*width+right], f2l = frame2[row*width+left];
            float f1d = frame1[down*width+col],  f1u = frame1[up*width+col];
            float f2d = frame2[down*width+col],  f2u = frame2[up*width+col];

            Ix[idx] = ((f1r - f1l) + (f2r - f2l)) * 0.25f * inv;
            Iy[idx] = ((f1d - f1u) + (f2d - f2u)) * 0.25f * inv;
            It[idx] = ((float)frame2[idx] - (float)frame1[idx]) * inv;
        }
    }

    // start from zero flow, then move between two buffers (like the GPU)
    std::vector<float> u_in(n, 0.0f), v_in(n, 0.0f);
    std::vector<float> u_out(n, 0.0f), v_out(n, 0.0f);

    for (int it = 0; it < iterations; it++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int left  = clampi(col - 1, 0, width  - 1);
                int right = clampi(col + 1, 0, width  - 1);
                int up    = clampi(row - 1, 0, height - 1);
                int down  = clampi(row + 1, 0, height - 1);
                int idx = row * width + col;

                float u_avg = (u_in[row*width+left] + u_in[row*width+right]
                             + u_in[up*width+col]   + u_in[down*width+col]) * 0.25f;
                float v_avg = (v_in[row*width+left] + v_in[row*width+right]
                             + v_in[up*width+col]   + v_in[down*width+col]) * 0.25f;

                float ix = Ix[idx], iy = Iy[idx], itv = It[idx];
                float P = (ix*u_avg + iy*v_avg + itv) / (alpha_sq + ix*ix + iy*iy);
                u_out[idx] = u_avg - ix * P;
                v_out[idx] = v_avg - iy * P;
            }
        }
        std::swap(u_in, u_out);   // this output is for the next iteratin
        std::swap(v_in, v_out);
    }

    // after the last swap the result lives in u_in / v_in
    std::copy(u_in.begin(), u_in.end(), u);
    std::copy(v_in.begin(), v_in.end(), v);
}


// Time CPU vs GPU on the same input then print the speedup
void benchmark_solver_cpu_vs_gpu(
    const unsigned char* d_gray1, const unsigned char* d_gray2,
    int width, int height,
    float alpha, int iterations)
{
    int n = width * height;

    // bring the grayscale frames to host (setup, not timed)
    std::vector<unsigned char> h_gray1(n), h_gray2(n);
    CUDA_CHECK(cudaMemcpy(h_gray1.data(), d_gray1, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gray2.data(), d_gray2, n, cudaMemcpyDeviceToHost));

    // CPU run, timed with chrono (averaged over CPU_RUNS)
    std::vector<float> u_cpu(n), v_cpu(n);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < CPU_RUNS; r++) {
        horn_schunck_cpu(h_gray1.data(), h_gray2.data(),
                         u_cpu.data(), v_cpu.data(),
                         width, height, alpha, iterations);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / CPU_RUNS;

    // GPU run, timed with cudaEvent
    float *d_u, *d_v;
    CUDA_CHECK(cudaMalloc(&d_u, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(float)));

    
    // GPU warmup
    horn_schunck(d_gray1, d_gray2, d_u, d_v, width, height, alpha, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());

    // timed run (reset the initial flow to zero again for a fair result)
    CUDA_CHECK(cudaMemset(d_u, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v, 0, n * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < GPU_RUNS; r++) {
        // reset to zero each run so every run does the same full work as the CPU
        CUDA_CHECK(cudaMemset(d_u, 0, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_v, 0, n * sizeof(float)));
        horn_schunck(d_gray1, d_gray2, d_u, d_v, width, height, alpha, iterations);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpu_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_total_ms, start, stop));
    float gpu_ms = gpu_total_ms / GPU_RUNS;

    // results
    std::vector<float> u_gpu(n), v_gpu(n);
    CUDA_CHECK(cudaMemcpy(u_gpu.data(), d_u, n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_gpu.data(), d_v, n*sizeof(float), cudaMemcpyDeviceToHost));
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        max_diff = std::max(max_diff, fabsf(u_cpu[i] - u_gpu[i]));
        max_diff = std::max(max_diff, fabsf(v_cpu[i] - v_gpu[i]));
    }

    printf("\n=== CPU vs GPU (Horn-Schunck solver, single level, %d iters) ===\n", iterations);
    printf("CPU time:   %.2f ms\n", cpu_ms);
    printf("GPU time:   %.2f ms\n", gpu_ms);
    printf("Speedup:    %.1fx\n", cpu_ms / gpu_ms);
    printf("Max CPU vs GPU flow diff: %.6f (should be ~0)\n\n", max_diff);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_u);
    cudaFree(d_v);
}

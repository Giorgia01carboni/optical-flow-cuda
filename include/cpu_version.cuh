#pragma once

// CPU (sequential) Horn-Schunck solver, single resolution.
// Used only as a baseline to measure the GPU speedup. It mirrors the GPU
// kernels (same gradients, same Jacobi update) so the two results match.

void horn_schunck_cpu(
    const unsigned char* frame1, const unsigned char* frame2,
    float* u, float* v,
    int width, int height,
    float alpha, int iterations);

// Runs the solver on CPU and on GPU on the same frames, times both, checks
// they agree, and prints the speedup. Single level, no pyramid.
void benchmark_solver_cpu_vs_gpu(
    const unsigned char* d_gray1, const unsigned char* d_gray2,
    int width, int height,
    float alpha, int iterations);

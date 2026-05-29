#pragma once
#include <vector>

struct PyramidLevel {
    unsigned char* d_frame1;  // grayscale, on GPU
    unsigned char* d_frame2;
    int width;
    int height;
};

// builds Gaussian pyramid of grayscale frame pairs 
// reminder: caller must free with free_pyramid()!!
std::vector<PyramidLevel> build_pyramid(
    const unsigned char* d_gray1,
    const unsigned char* d_gray2,
    int width, int height,
    int num_levels   
);

void free_pyramid(std::vector<PyramidLevel>& pyramid);

// 1. runs Horn-Schunck across all pyramid levels. 2. coarse to fine
// 3. writes result into d_u, d_v (full resolution, caller allocates)
void pyramidal_horn_schunck(
    const unsigned char* d_gray1,
    const unsigned char* d_gray2,
    float* d_u, float* d_v,
    int width, int height,
    float alpha,
    int iterations_per_level,
    int num_levels
);
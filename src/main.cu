#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "cuda_utils.cuh"
#include "horn_schunck.cuh"
#include "pyramid.cuh"
#include "warp.cuh"
#include "blend.cuh"
#include "metrics.cuh"
#include "cpu_version.cuh"

#include <cstring>
#include <cstdio>
#include <cmath>   

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

// Colored flow, for debugging
__device__ void hsv_to_rgb(float h, float s, float v,
                           float& r, float& g, float& b) {
    float c = v * s;
    float hp = h / 60.0f;
    float x = c * (1.0f - fabsf(fmodf(hp, 2.0f) - 1.0f));
    float r1=0,g1=0,b1=0;
    if (hp < 1)      { r1=c; g1=x; }
    else if (hp < 2) { r1=x; g1=c; }
    else if (hp < 3) { g1=c; b1=x; }
    else if (hp < 4) { g1=x; b1=c; }
    else if (hp < 5) { r1=x; b1=c; }
    else             { r1=c; b1=x; }
    float m = v - c;
    r = r1+m; g = g1+m; b = b1+m;
}

__global__ void flow_to_color(
    const float* __restrict__ u, const float* __restrict__ v,
    unsigned char* __restrict__ out, int width, int height,
    float max_mag)   // magnitudini >= max_mag saturano
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    int idx = row * width + col;

    float fu = u[idx], fv = v[idx];
    float mag = sqrtf(fu*fu + fv*fv);
    float hue = (atan2f(fv, fu) + 3.14159265f) * (180.0f / 3.14159265f); // [0,360)
    float sat = fminf(mag / max_mag, 1.0f);

    float r,g,b;
    hsv_to_rgb(hue, sat, 1.0f, r, g, b);
    out[idx*3+0] = (unsigned char)(r*255.0f);
    out[idx*3+1] = (unsigned char)(g*255.0f);
    out[idx*3+2] = (unsigned char)(b*255.0f);
}

int main(int argc, char** argv)
{
    // Just parsing configurations. Expected usage: ./optical_flow_cuda data/[dataset_name] [num_levels]
    if (argc < 2 || argc > 3) {
        printf("Usage: %s <data_folder> [num_levels]\n", argv[0]);
        printf("  expects <data_folder>/frame10.png, frame11.png, and optional flow10.flo\n");
        printf("  outputs go to data/results/\n");
        return 1;
    }
    const char* data_folder = argv[1];
    int num_levels = (argc == 3) ? atoi(argv[2]) : 4;
    if (num_levels < 1) num_levels = 1;
    printf("Pyramid levels: %d\n", num_levels);

    // derive the dataset name from the folder path 
    char dataset_name[256];
    {
        const char* slash = strrchr(data_folder, '/');   // last '/' in the path
        const char* name  = slash ? slash + 1 : data_folder;
        strncpy(dataset_name, name, sizeof(dataset_name) - 1);
        dataset_name[sizeof(dataset_name) - 1] = '\0';
        // remove a trailing slash if the user passed ".../RubberWhale/"
        size_t len = strlen(dataset_name);
        if (len > 0 && dataset_name[len - 1] == '/') dataset_name[len - 1] = '\0';
    }

    // --- build the input paths from the folder ---
    char path1[512], path2[512], gt_buf[512];
    snprintf(path1, sizeof(path1), "%s/frame10.png", data_folder);
    snprintf(path2, sizeof(path2), "%s/frame11.png", data_folder);
    snprintf(gt_buf, sizeof(gt_buf), "%s/flow10.flo", data_folder);

    // ground truth is optional: use it only if the file actually exists
    const char* gt_path = nullptr;
    {
        FILE* probe = fopen(gt_buf, "rb");
        if (probe) { fclose(probe); gt_path = gt_buf; }
    }

    // interpolated frame goes into data/results/<dataset>_interp.png 
    char out_path[512];
    snprintf(out_path, sizeof(out_path), "data/results/%s_interp.png", dataset_name);

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

    // Check cpu-gpu benchmark
    benchmark_solver_cpu_vs_gpu(d_gray1, d_gray2, width, height, 0.5f, 300);

    // optical flow between the two frames
    // flow buffers —> one float per pixel per direction
    float *d_u, *d_v;
    CUDA_CHECK(cudaMalloc(&d_u, n_gray * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, n_gray * sizeof(float)));

    // run Horn-Schunck: alpha=10 (smoothness), 100 iterations REMEMBER TO CHECK THIS AGAIN
    pyramidal_horn_schunck(d_gray1, d_gray2, d_u, d_v, width, height, 0.5f, 300, num_levels);

    printf("Pyramidal Horn-Schunck computed\n");

    {
        unsigned char* d_flowviz;
        CUDA_CHECK(cudaMalloc(&d_flowviz, n_gray));
        flow_to_image<<<grid, block>>>(d_u, d_v, d_flowviz, width, height, 10.0f);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned char* h_flowviz = (unsigned char*)malloc(n_gray);
        CUDA_CHECK(cudaMemcpy(h_flowviz, d_flowviz, n_gray, cudaMemcpyDeviceToHost));

        char flow_path[512];
        snprintf(flow_path, sizeof(flow_path), "data/results/%s_flow.png", dataset_name);
        stbi_write_png(flow_path, width, height, 1, h_flowviz, width);
        printf("Saved flow: %s\n", flow_path);

        cudaFree(d_flowviz);
        free(h_flowviz);
    }

    // Apply color coding to flow for better visualization of direction and magnitude (direction = hue, magnitude = saturation)
    {
        unsigned char* d_flowcol;
        CUDA_CHECK(cudaMalloc(&d_flowcol, n_rgb));
        flow_to_color<<<grid, block>>>(d_u, d_v, d_flowcol, width, height, 20.0f);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned char* h_flowcol = (unsigned char*)malloc(n_rgb);
        CUDA_CHECK(cudaMemcpy(h_flowcol, d_flowcol, n_rgb, cudaMemcpyDeviceToHost));

        char colpath[512];
        snprintf(colpath, sizeof(colpath), "data/results/%s_flowcolor.png", dataset_name);
        stbi_write_png(colpath, width, height, 3, h_flowcol, width * 3);
        printf("Saved flow color: %s\n", colpath);

        cudaFree(d_flowcol);
        free(h_flowcol);
    }

    if (gt_path) {
        evaluate_average_endpoint_error(d_u, d_v, gt_path, width, height);
    }

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
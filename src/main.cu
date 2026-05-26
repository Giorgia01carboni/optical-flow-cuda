#include <cstdio>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA device(s)\n", deviceCount);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device 0: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Global memory: %.1f GB\n", prop.totalGlobalMem / 1e9);

    return 0;
}

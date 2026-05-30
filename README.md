# Optical Flow CUDA

A GPU-accelerated optical flow and frame interpolation pipeline, written from
scratch in CUDA / C++. It estimates dense motion between two frames with a
**pyramidal Horn-Schunck** solver, then uses that motion to synthesise an
in-between frame.

The goal of the project is twofold: build a working interpolation pipeline, and
show a clear speedup of the parallel GPU solver over a sequential CPU baseline.

## Example

| Frame 1 | Frame 2 | Interpolated (ours) |
|---------|---------|---------------------|
| ![frame1](data/RubberWhale/frame10.png) | ![frame2](data/RubberWhale/frame11.png) | ![result](data/results/RubberWhale_interp.png) |

## Pipeline

1. **Optical flow** (`flow/`): pyramidal Horn-Schunck. The pyramid handles large
   motion; each level runs an iterative Jacobi solver. Intensities are
   normalized to [0,1] and smoothness is controlled by `alpha`.
2. **Warping** (`interp/warp.cu`): move pixels along the flow to a mid-point in time.
3. **Blending** (`interp/blend.cu`): merge the two warped frames into the final result.

## Structure

```
optical-flow-cuda/
├── CMakeLists.txt
├── include/              # headers (.cuh)
│   ├── horn_schunck.cuh
│   ├── pyramid.cuh
│   ├── warp.cuh
│   ├── blend.cuh
│   ├── metrics.cuh       # .flo reader + EPE (parallel reduction)
│   ├── cpu_version.cuh   # CPU baseline for benchmarking
│   ├── cuda_utils.cuh
│   └── stb_image*.h      # third-party image I/O
├── src/
│   ├── main.cu           # orchestration
│   ├── flow/             # horn_schunck.cu, pyramid.cu
│   ├── interp/           # warp.cu, blend.cu
│   └── utils/            # metrics.cu, cpu_version.cu
└── data/
    ├── RubberWhale/      # frame10.png, frame11.png, flow10.flo
    └── results/          # generated output
```

## Build

Requires CUDA and CMake. Built and tested on an NVIDIA RTX 4090 (compute 8.9),
CUDA 13.0.

```bash
mkdir -p build
cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8 && cd ..
```

`Release` enables `-O3`. This matters: the CPU baseline must be optimized,
otherwise the measured speedup is not fair.

## Run

Run from the project root. Pass a data folder and the number of pyramid levels.
The folder must contain `frame10.png` and `frame11.png`, and optionally
`flow10.flo` (ground truth). Output is written to `data/results/`.

```bash
# 2 pyramid levels (good for small motion)
./build/optical_flow data/RubberWhale 2
```

For each run the program saves the interpolated frame, a grayscale flow image,
a color flow image, and, if a `.flo` is present, prints the Average Endpoint
Error (EPE). It also prints a CPU vs GPU speedup for the solver.

## Dataset

Tested on the [Middlebury optical flow benchmark](https://vision.middlebury.edu/flow/data/)
(the "Other Datasets" with public ground-truth flow), mainly **RubberWhale** and
**Dimetrodon**. These have small, smooth motion and dense texture, which suits
Horn-Schunck well.

## Results (RubberWhale, 2 levels)

| Metric | Value |
|--------|-------|
| Average EPE | 0.61 px |
| GPU solver time | ~1.4 ms |
| CPU solver time | ~131 ms |
| Speedup | ~94x |

The speedup is for the single-level iterative solver (compute only, no host
to device transfers). CPU baseline is single-threaded, compiled with `-O3`;
GPU time is averaged over 100 runs after a warmup. The CPU and GPU flow fields
match exactly (max difference 0.000000), which confirms the GPU result is correct.

## Notes

Horn-Schunck is a classic method and works best on gentle scenes. On hard
real-world footage (large motion, parallax, reflective surfaces) the result
degrades, as expected for this class of algorithm.

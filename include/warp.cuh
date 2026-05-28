#pragma once

// warps src frame using flow field (d_u, d_v) scaled by t
// output is an RGB image (3 bytes per pixel)
// t=0.5 means halfway, t=1.0 means full flow
void warp_frame(
    const unsigned char* d_src,   // source RGB frame (on GPU)
    const float* d_u,             // horizontal flow
    const float* d_v,             // vertical flow
    unsigned char* d_out,         // output RGB frame (caller allocates)
    int width, int height,
    float t                       // interpolation factor 0.0 to 1.0
);
#pragma once

// blends two RGB frames 50/50 into output
void blend_frames(
    const unsigned char* d_frame_a,
    const unsigned char* d_frame_b,
    unsigned char* d_out,
    int width, int height
);
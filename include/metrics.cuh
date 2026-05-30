#pragma once

// Flow evaluation: compare our estimated optical flow against a ground-truth
// .flo file (Middlebury format) and measure how wrong we are.

// The metric is the Average Endpoint Error (EPE): for every pixel we take the
// distance between our motion vector (u,v) and the true one (u_gt,v_gt), then
// average over all valid pixels. Lower = better. 

// Load a Middlebury .flo ground-truth file into CPU (host) memory.

// HOW THE DATA IS STORED IN .FLO
// Flow has two numbers per pixel: u (horizontal motion) and v (vertical motion).
// For a W×H image that means W*H pairs (u,v). Middlebury stores them interleaved:
// u and v of the SAME pixel sit next to each other, then the next pixel follows.

//   [u0, v0, u1, v1, u2, v2, ...]
//   =[px 0    px 1    px 2]

// So pixel number i has its u at position (2*i) and its v at position (2*i + 1).
// The returned array therefore has W*H*2 floats in total.


// Returns: pointer to the interleaved array, or nullptr on error.
float* load_groundtruth_flow(const char* path,
                             int* out_width, int* out_height);


// Compute the Average Endpoint Error between our flow (d_u, d_v, already on the
// GPU) and the ground truth stored at gt_path.
//
// Returns: the average EPE in pixels, or -1.0 on error (file unreadable, or its
// size does not match our flow). Also prints a short diagnostic line.
double evaluate_average_endpoint_error(const float* d_u, const float* d_v,
                                       const char* gt_path,
                                       int width, int height);

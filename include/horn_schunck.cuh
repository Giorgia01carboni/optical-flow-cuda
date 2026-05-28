#pragma once

/* Horn-Schunck makes two assumptions:
1. Brightness constancy: the intensity of a pixel doesn't change between frames:
    Ix·u + Iy·v + It = 0 where Ix, Iy, It are the spatial and temporal gradients of the image, and u, v are the flow vectors we want to solve for.

2. Smoothness: the flow field is smooth (neighboring pixels have similar flow)
ū  = average of u in the 4-pixel neighbourhood
v̄  = average of v in the 4-pixel neighbourhood
P  = (Ix·ū + Iy·v̄ + It) / (α² + Ix² + Iy²)

u_new = ū - Ix · P
v_new = v̄ - Iy · P

The algorithm iteratively updates u and v using the above equations, where α is a regularization parameter that controls the smoothness of the flow field. 
The iterations continue until convergence or until a specified number of iterations is reached.
*/

// runs Horn-Schunck on two grayscale frames already on the GPU
// outputs d_u, d_v (caller must allocate them: width*height*sizeof(float))
void horn_schunck(
    const unsigned char* d_frame1,
    const unsigned char* d_frame2,
    float* d_u,
    float* d_v,
    int width, int height,
    float alpha,       // smoothness weight
    int iterations     // how many Jacobi steps, I'll try 100
);
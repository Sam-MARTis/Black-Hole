#include "kernel.hpp"
#include <iostream>
#define LIMIT 100
#define STEP 0.1f

#define MAGNITUDE_SQ(x, y, z) ((x)*(x) + (y)*(y) + (z)*(z))

#define TX 32
#define TY 32

#define HALF_FOV 0.5235 
// Half


__global__
void ray_tracing(uchar4 *d_out, float ds, float3 right, float3 up, float3 forward, int width, int height){
    
    const int c = blockDim.x*blockIdx.x + threadIdx.x;
    const int r = blockDim.y*blockIdx.y + threadIdx.y;
    if (c >= width || r >= height) return;

    const int xidx = c - width*0.5f;
    const int yidx = r - height*0.5f;

    const float t_right = HALF_FOV * xidx*2.0f/width;
    const float t_up =    HALF_FOV * yidx*2.0f/height;

    // const float sin_right = t_right - t_right*t_right*t_right*0.3333f;
    // // const float cos_right = 1.0f - sin_right*sin_right*0.5f;
    // const float sin_up = t_up - t_up*t_up*t_up*0.3333f;
    // const float cos_up = sqrtf(1.0f - sin_up*sin_up);

    const float sin_right = sinf(t_right);
    const float cos_right = cosf(t_right);
    const float sin_up = sinf(t_up);
    const float cos_up = cosf(t_up);

    const float mag_inv = 1/(1+ sin_right*sin_right + sin_up*sin_up);
    const float3 direction = { (forward.x + sin_right*right.x + sin_up*up.x) * mag_inv, (forward.y + sin_right*right.y + sin_up*up.y) * mag_inv, (forward.z + sin_right*right.z + sin_up*up.z) * mag_inv };
    const float3 step = {direction.x * STEP, direction.y * STEP, direction.z * STEP};

    const int idx = r*width + c;
    float3 pos = {-forward.x + ds*(right.x*xidx + up.x*yidx), -forward.y + ds*(right.y*xidx + up.y*yidx), -forward.z + ds*(right.z*xidx + up.z*yidx)};
    
    float theta = 0.0f;
    float phi = 0.0f;

    // Direction and position sorted. Phew

    char ray_colour = 0;
    char colours[3*3] = {0, 0, 0, 254, 254, 254, 254, 0, 0};

    const int max_iterations = LIMIT / STEP;
    for(int i = 0; i < max_iterations; i++){
        pos.x += step.x;
        pos.y += step.y;
        pos.z += step.z;

        if (MAGNITUDE_SQ(pos.x, pos.y, pos.z) < 0.05f) {
            ray_colour = 1;
            break;
        }
        if (MAGNITUDE_SQ(pos.x - 0.8f, pos.y - 0.3f, pos.z) < 0.05f) {
            ray_colour = 2;
            break;
        }
    }
    d_out[idx].x = colours[3*ray_colour + 0];
    d_out[idx].y = colours[3*ray_colour + 1];
    d_out[idx].z = colours[3*ray_colour + 2];
    d_out[idx].w = 255;

    // d_out[idx].x = 255; 
    // d_out[idx].y = 255; 
    // d_out[idx].z = 255; 
    // d_out[idx].w = 255; 
    
    

    return;
}


float3 cross_product(float3 a, float3 b){
    float3 c = {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x*b.y - a.y*b.x};
    return c;
}
float3 normalize_vector(float3 a){
    float mag_inv = 1/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
    float3 b = {a.x*mag_inv, a.y*mag_inv, a.z*mag_inv};   
    return b;
}

__global__
void testKernel(uchar4 *d_out, int width, int height){
    int xidx = blockDim.x*blockIdx.x + threadIdx.x - width/2;
    int yidx = blockDim.y*blockIdx.y + threadIdx.y - height/2;

    const int c = blockDim.x*blockIdx.x + threadIdx.x;
    const int r = blockDim.y*blockIdx.y + threadIdx.y;
    
    // Bounds checking
    if (c >= width || r >= height) return;
    
    const int idx = r*width + c;

    d_out[idx].x = 255;
    d_out[idx].y = 255;
    d_out[idx].z = 255;
    d_out[idx].w = 255;
}

void testKernelLauncher(uchar4 *d_out, int width, int height){

    const dim3 blockSize(TX, TY);
    const dim3 gridSize((width + TX - 1) / TX, (height + TY - 1) / TY);
    
    testKernel<<<gridSize, blockSize>>>(d_out, width, height);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();

}

void kernelLauncher(uchar4 *d_out, float camerax, float cameray, float cam_distance,  int width, int height,float scaling, int count, float *spheres_coords, float* spheres_rads){

    float3 up = {0.0f ,0.0f, 1.0f};
    float sinphi = sinf(cameray);
    float cosphi = cosf(cameray);
    float sintheta = sinf(camerax);
    float costheta = cosf(camerax);
    // printf("Camera: %f %f\n", camerax, cameray);

    float3 r = {costheta*sinphi, sintheta*sinphi, cosphi};
    float3 plane_right = normalize_vector(cross_product(up, r));
    float3 plane_up = normalize_vector(cross_product(r, plane_right));
    float3 plane_normal = {-r.x, -r.y, -r.z};

    const dim3 blockSize(TX, TY);
    const dim3 gridSize((width + TX - 1) / TX, (height + TY - 1) / TY);
    

    ray_tracing<<<gridSize, blockSize>>>(d_out, scaling, plane_right, plane_up, plane_normal, width, height);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();

    


    return;
}
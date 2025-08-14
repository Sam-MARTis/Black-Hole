#include "kernel.hpp"
#include <iostream>
#define LIMIT 10
#define STEP 0.05f

#define MAGNITUDE_SQ(x, y, z) ((x)*(x) + (y)*(y) + (z)*(z))

#define TY 16
#define TX 16
#define SCHWARZCHILD_RADIUS 0.2f

#define HALF_FOV 0.5235 
// Half


__global__
void ray_tracing(uchar4 *d_out, float ds, float3 right, float3 up, float3 forward, int width, int height){
    
    const int col = blockDim.x*blockIdx.x + threadIdx.x;
    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    if (col >= width || row >= height) return;

    const int xidx = col - width*0.5f;
    const int yidx = row - height*0.5f;

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
    // const float3 step = {direction.x * STEP, direction.y * STEP, direction.z * STEP};

    const int idx = row*width + col;


    float3 pos = {-forward.x + ds*(right.x*xidx + up.x*yidx), -forward.y + ds*(right.y*xidx + up.y*yidx), -forward.z + ds*(right.z*xidx + up.z*yidx)};

    float r = sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);

    float theta = acosf(pos.z / r);
    float phi = atan2f(pos.y, pos.x);
    float dx = direction.x;
    float dy = direction.y;
    float dz = direction.z;

    float dr     = sin(theta)*cos(phi)*dx + sin(theta)*sin(phi)*dy + cos(theta)*dz;
    float dtheta = (cos(theta)*cos(phi)*dx + cos(theta)*sin(phi)*dy - sin(theta)*dz) / r;
    float dphi   = (-sin(phi)*dx + cos(phi)*dy) / (r * sin(theta));

    float L = r*r*sin(theta)*dphi;
    float f = 1.0f - SCHWARZCHILD_RADIUS/r;
    float dt_dL = sqrt((dr*dr)/f + r*r*(dtheta*dtheta + sin(theta)*sin(theta)*dphi*dphi));
    float E = f * dt_dL;



    
    char ray_colour = 0;
    char colours[3*3] = {0, 0, 0, 254, 254, 254, 254, 0, 0};


    const int max_iterations = LIMIT / STEP;
    for(int i = 0; i < max_iterations; i++){
        // pos.x += step.x;
        // pos.y += step.y;
        // pos.z += step.z;

        const float ddr = - (SCHWARZCHILD_RADIUS / (2.0 * r*r)) * f * dt_dL * dt_dL
         + (SCHWARZCHILD_RADIUS / (2.0 * r*r * f)) * dr * dr
         + r * (dtheta*dtheta + sin(theta)*sin(theta)*dphi*dphi);

        const float ddphi =  -2.0*dr*dphi/r - 2.0*cos(theta)/(sin(theta)) * dtheta * dphi;
        const float ddtheta = -2.0*dr*dtheta/r + sin(theta)*cos(theta)*dphi*dphi;


        r += dr * STEP;
        theta += dtheta * STEP;
        phi += dphi * STEP;
        dr += ddr * STEP;
        dtheta += ddtheta * STEP;
        dphi += ddphi * STEP;

        pos.x = r * sin(theta) * cos(phi);
        pos.y = r * sin(theta) * sin(phi);
        pos.z = r * cos(theta);
        

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
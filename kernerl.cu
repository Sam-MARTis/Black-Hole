#include "kernel.hpp"
#define LIMIT 10

#define TX 32
#define TY 32

void ray_tracing(uchar4 *d_out, float ds, float3 right, float3 up, float3 forward, int width, int height){
    int xidx = blockDim.x*blockIdx.x + threadIdx.x - width/2;
    int yidx = blockDim.y*blockIdx.y + threadIdx.y - height/2;

    const int c = blockDim.x*blockIdx.x + threadIdx.x;
    const int r = blockDim.y*blockIdx.y + threadIdx.y;
    const int idx = r*width + c;
    float3 pos = {-forward.x + ds*(right.x*xidx + up.x*yidx), -forward.y + ds*(right.y*xidx + up.y*yidx), -forward.z + ds*(right.z*xidx + up.z*yidx)};

    bool infty_ray = true;
    
    uchar4& cell = d_out[idx];
    cell.x = 255;
    cell.y = 0;
    cell.z = 0;
    cell.w = 0;
    

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



void kernelLauncher(uchar4 *d_out, float2 camera, float cam_distance,  int width, int height,float scaling, int count, float *spheres_coords, float* spheres_rads){

    float3 up = {0.0f ,0.0f, 1.0f};
    float sinphi = sinf(camera.y);
    float cosphi = cosf(camera.y);
    float sintheta = sinf(camera.x);
    float costheta = cosf(camera.x);

    float3 r = {costheta*sinphi, sintheta*sinphi, cosphi};
    float3 plane_right = cross_product(up, r);
    float3 plane_up = cross_product(r, plane_right);
    float3 plane_normal = {-r.x, -r.y, -r.z};

    // ray_tracing<<
    const dim3 blockSize(TX, TY);
    const dim3 gridSize((int)ceil(width/TX), (int)ceil(height/TY));
    ray_tracing<<<gridSize, blockSize>>>(d_out, scaling, plane_right, plane_up, plane_normal, width, height);

    


    return;
}
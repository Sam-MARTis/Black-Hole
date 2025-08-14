#include "kernel.hpp"
#include <iostream>
#define LIMIT 5
#define STEP 0.01f
#define SCREEN_DISTANCE 1.0f

#define MAGNITUDE_SQ(x, y, z) ((x)*(x) + (y)*(y) + (z)*(z))

#define TY 16
#define TX 32
#define SCHWARZCHILD_RADIUS 0.2f

#define ACCRETION_RADIUS 0.5f
#define ACCRETION_DPHI_SQ 0.025f

#define HALF_FOV 0.5235 
// Half

__device__ void getGeodesicDerivatives(const float r, const float theta, const float phi, const float dr, float dtheta, float dphi, const float f, const float dt_dL, float out[6]) {
    

    out[0] = dr; 
    out[1] = dtheta; 
    out[2] = dphi; 

    float sintheta = sinf(theta);
    float costheta = cosf(theta);
    float rinv = 1.0f / r;
    out[3] = - (SCHWARZCHILD_RADIUS *0.5f * rinv*rinv) * f * dt_dL * dt_dL
         + (SCHWARZCHILD_RADIUS / (2.0 * r*r * f)) * dr * dr
         + r * (dtheta*dtheta + sintheta*sintheta*dphi*dphi);
    out[4] = -2.0*dr*dtheta*rinv + sintheta*costheta*dphi*dphi;
    out[5] = -2.0*dr*dphi*rinv - 2.0*costheta/(sintheta) * dtheta * dphi;

}
__global__
void ray_tracing(uchar4 *d_out, float ds, float3 right, float3 up, float3 forward, float r0, int width, int height){
    
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

    const float sin_right = t_right - t_right*t_right*t_right*0.3333f;
    // const float cos_right = cosf(t_right);
    const float cos_right = 1.0f - t_right*t_right*0.5f;
    const float sin_up = t_up - t_up*t_up*t_up*0.3333f;
    const float cos_up = 1.0f - sin_up*sin_up*0.5f;

    const float mag_inv = 1/(1+ sin_right*sin_right + sin_up*sin_up);
    // const float3 direction = { (forward.x + sin_right*right.x + sin_up*up.x) * mag_inv, (forward.y + sin_right*right.y + sin_up*up.y) * mag_inv, (forward.z + sin_right*right.z + sin_up*up.z) * mag_inv };
    // const float3 step = {direction.x * STEP, direction.y * STEP, direction.z * STEP};

    const int idx = row*width + col;


    float3 pos = {(-forward.x + ds*(right.x*xidx + up.x*yidx))*r0, (-forward.y + ds*(right.y*xidx + up.y*yidx))*r0, -forward.z + ds*(right.z*xidx + up.z*yidx)*r0};

    float r = sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
    float rinv = 1.0f / r;

    float theta = acosf(pos.z * rinv);
    float phi = atan2f(pos.y, pos.x);
    float dx = (forward.x + sin_right*right.x + sin_up*up.x) * mag_inv;
    float dy = (forward.y + sin_right*right.y + sin_up*up.y) * mag_inv;
    float dz = (forward.z + sin_right*right.z + sin_up*up.z) * mag_inv;

    const float sintheta = sinf(theta);
    const float costheta = cosf(theta);
    const float sinphi = sinf(phi);
    const float cosphi = cosf(phi);

    float dr     = sintheta*cosphi*dx + sintheta*sinphi*dy + costheta*dz;
    float dtheta = (costheta*cosphi*dx + costheta*sinphi*dy - sintheta*dz) * rinv;
    float dphi   = (-sinphi*dx + cosphi*dy) / (r * sintheta);

    const float L = r*r*sintheta*dphi;
    const float f = 1.0f - SCHWARZCHILD_RADIUS * rinv;
    const float dt_dL = sqrtf((dr*dr)/f + r*r*(dtheta*dtheta + sintheta*sintheta*dphi*dphi));



    
    char ray_colour = 0;
    char colours[3*3] = {0, 0, 0, 255, 255, 255, 255, 0, 0};


    const int max_iterations = LIMIT / STEP;
    for(int i = 0; i < max_iterations; i++){
        // pos.x += step.x;
        // pos.y += step.y;
        // pos.z += step.z;

        // const float ddr = - (SCHWARZCHILD_RADIUS / (2.0 * r*r)) * f * dt_dL * dt_dL
        //  + (SCHWARZCHILD_RADIUS / (2.0 * r*r * f)) * dr * dr
        //  + r * (dtheta*dtheta + sin(theta)*sin(theta)*dphi*dphi);

        // const float ddphi =  -2.0*dr*dphi/r - 2.0*cos(theta)/(sin(theta)) * dtheta * dphi;
        // const float ddtheta = -2.0*dr*dtheta/r + sin(theta)*cos(theta)*dphi*dphi;

        float k1[6], k2[6], k3[6], k4[6];
        const float halfstep = 0.5f * STEP;
        getGeodesicDerivatives(r, theta, phi, dr, dtheta, dphi, f, dt_dL, k1);
        getGeodesicDerivatives(r + halfstep*k1[0], theta + halfstep*k1[1], phi + halfstep*k1[2], dr + halfstep*k1[3], dtheta + halfstep*k1[4], dphi + halfstep*k1[5], f, dt_dL, k2);
        getGeodesicDerivatives(r + halfstep*k2[0], theta + halfstep*k2[1], phi + halfstep*k2[2], dr + halfstep*k2[3], dtheta + halfstep*k2[4], dphi + halfstep*k2[5], f, dt_dL, k3);
        getGeodesicDerivatives(r + STEP*k3[0], theta + STEP*k3[1], phi + STEP*k3[2], dr + STEP*k3[3], dtheta + STEP*k3[4], dphi + STEP*k3[5], f, dt_dL, k4);    
    
        const float onesixth = STEP / 6.0f;
        r += onesixth * (k1[0] + 2.0f*k2[0] + 2.0f*k3[0] + k4[0]);
        theta += onesixth * (k1[1] + 2.0f*k2[1] + 2.0f*k3[1] + k4[1]);
        phi += onesixth * (k1[2] + 2.0f*k2[2] + 2.0f*k3[2] + k4[2]);
        dr += onesixth * (k1[3] + 2.0f*k2[3] + 2.0f*k3[3] + k4[3]);
        dtheta += onesixth * (k1[4] + 2.0f*k2[4] + 2.0f*k3[4] + k4[4]);
        dphi += onesixth * (k1[5] + 2.0f*k2[5] + 2.0f*k3[5] + k4[5]);   
        // r += out[0] * STEP;
        // theta += out[1] * STEP;
        // phi += out[2] * STEP;
        // dr += out[3] * STEP;
        // dtheta += out[4] * STEP;
        // dphi += out[5] * STEP;

        float sintheta_this = sinf(theta);
  
        pos.x = r * sintheta_this * cosf(phi);
        pos.y = r * sintheta_this * sinf(phi);
        pos.z = r * cosf(theta);


        if (r < SCHWARZCHILD_RADIUS) {
            ray_colour = 1;
            break;
        }
        // if (MAGNITUDE_SQ(pos.x - 0.8f, pos.y - 0.3f, pos.z) < 0.01f) {
        //     ray_colour = 2;
        //     break;
        // }
        // if((r < ACCRETION_RADIUS)){
        //     const float phi_sq = phi*phi;
        //     if(phi_sq > (3.1415f - ACCRETION_DPHI_SQ)*(3.1415f - ACCRETION_DPHI_SQ) && phi_sq < (3.1415f + ACCRETION_DPHI_SQ)*(3.1415f + ACCRETION_DPHI_SQ) || (phi_sq < ACCRETION_DPHI_SQ)){
        //         ray_colour = 2;
        //         break;
        //     }

        // }
        if(phi < 0.001f && phi > -0.001f){
            ray_colour = 2;
            break;
        }
        if(phi > 3.141f - 0.001f && phi < 3.141f + 0.001f){
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

// __global__
// void testKernel(uchar4 *d_out, int width, int height){
//     int xidx = blockDim.x*blockIdx.x + threadIdx.x - width/2;
//     int yidx = blockDim.y*blockIdx.y + threadIdx.y - height/2;

//     const int c = blockDim.x*blockIdx.x + threadIdx.x;
//     const int r = blockDim.y*blockIdx.y + threadIdx.y;
    
//     // Bounds checking
//     if (c >= width || r >= height) return;
    
//     const int idx = r*width + c;

//     d_out[idx].x = 255;
//     d_out[idx].y = 255;
//     d_out[idx].z = 255;
//     d_out[idx].w = 255;
// }

// void testKernelLauncher(uchar4 *d_out, int width, int height){

//     const dim3 blockSize(TX, TY);
//     const dim3 gridSize((width + TX - 1) / TX, (height + TY - 1) / TY);
    
//     testKernel<<<gridSize, blockSize>>>(d_out, width, height);
    
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
//     }
    
//     cudaDeviceSynchronize();

// }

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
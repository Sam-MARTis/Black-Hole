#include "kernel.hpp"
#include <iostream>
#define LIMIT 30
#define STEP 0.02f
#define SCREEN_DISTANCE 1.0f

#define MAGNITUDE_SQ(x, y, z) ((x)*(x) + (y)*(y) + (z)*(z))

#define TY 16
#define TX 16
#define SCHWARZCHILD_RADIUS 0.3f

#define ACCRETION_RADIUS 0.9f
#define ACCRETION_DPHI 0.05f
#define PI 3.14159265358979323846f


#define HALF_FOV 0.4235 
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
    const float out_r0 = r0 *2.0f;
    const float inv_schwarzchild_radius = 1.0f / SCHWARZCHILD_RADIUS;
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

    // const float sin_right = t_right - t_right*t_right*t_right*0.3333f;
    const float sin_right = sinf(t_right);

    // const float cos_right = cosf(t_right);
    // const float cos_right = 1.0f - t_right*t_right*0.5f;
    // const float sin_up = t_up - t_up*t_up*t_up*0.3333f;
    const float sin_up = sinf(t_up);
    // const float cos_up = 1.0f - sin_up*sin_up*0.5f;

    // const float3 direction = { (forward.x + sin_right*right.x + sin_up*up.x) * mag_inv, (forward.y + sin_right*right.y + sin_up*up.y) * mag_inv, (forward.z + sin_right*right.z + sin_up*up.z) * mag_inv };
    // const float3 step = {direction.x * STEP, direction.y * STEP, direction.z * STEP};
    
    const int idx = row*width + col;
    
    
    float3 pos = {(-forward.x + ds*(right.x*xidx + up.x*yidx))*r0, (-forward.y + ds*(right.y*xidx + up.y*yidx))*r0, -forward.z + ds*(right.z*xidx + up.z*yidx)*r0};
    
    float r = sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
    float rinv = 1.0f / r;
    
    float theta = acosf(pos.z * rinv);
    float phi = atan2f(pos.y, pos.x);
    float dx = (forward.x + sin_right*right.x + sin_up*up.x);
    float dy = (forward.y + sin_right*right.y + sin_up*up.y);
    float dz = (forward.z + sin_right*right.z + sin_up*up.z);
    
    const float mag_inv = 1/(sqrtf(dx*dx + dy*dy + dz*dz));
    dx *= mag_inv;
    dy *= mag_inv;
    dz *= mag_inv;

    const float sintheta = sinf(theta);
    const float costheta = cosf(theta);
    const float sinphi = sinf(phi);
    const float cosphi = cosf(phi);

    float dr     = sintheta*cosphi*dx + sintheta*sinphi*dy + costheta*dz;
    float dtheta = (costheta*cosphi*dx + costheta*sinphi*dy - sintheta*dz) * rinv;
    float dphi   = (-sinphi*dx + cosphi*dy) / (r * sintheta);


    const float f = 1.0f - SCHWARZCHILD_RADIUS * rinv;
    const float dt_dL = sqrtf((dr*dr)/f + r*r*(dtheta*dtheta + sintheta*sintheta*dphi*dphi));



    
    char ray_colour = 0;
    char colours[3*3] = {20, 20, 20, 0, 0, 0, 255, 0, 0};
    
    
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

        // // }
        if(r<ACCRETION_RADIUS){
            if(pos.x*ACCRETION_RADIUS < 0.1f*(1.3f*ACCRETION_RADIUS - r) && pos.x*ACCRETION_RADIUS > -0.1f*(1.3f*ACCRETION_RADIUS - r))
            {
                // if(pos.x < ACCRETION_DPHI && pos.x > -ACCRETION_DPHI){
                    ray_colour = 2;
                    break;
                // }
            }
        // if(phi*r < ACCRETION_DPHI && phi*r > -ACCRETION_DPHI){
        //     ray_colour = 2;
        //     break;
        // }

        // if( ((phi+PI)*r < ACCRETION_DPHI && (phi+PI)*r > -ACCRETION_DPHI) || ((phi-PI)*r < ACCRETION_DPHI && (phi-PI)*r > -ACCRETION_DPHI) ){
        //     ray_colour = 2;
        //     break;
        // }
        }
        if(r> out_r0){
            // const float theta_divisions = 50.0f;
            // const float phi_divisions = 10.0f;
            // const float theta_step = 2*PI / theta_divisions;
            // const float phi_step = PI / phi_divisions;
            // const float theta_idx = floorf(theta / theta_step);
            // const float phi_idx = floorf(phi / phi_step);
            // const int colour_idx = (int)(phi_idx  );
            // char colour;
            // if(colour_idx & 1){
            //     colour = 255;
            // }
            // else{
            //     colour = 0;
            // }

            // d_out[idx].x = colour;
            // d_out[idx].y = colour;
            // d_out[idx].z = colour;
            // d_out[idx].w = 255;
            // return;
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


float3 cross_product(float3 a, float3 b, float scaling=1.0f){
    float3 c = {(a.y * b.z - a.z * b.y) * scaling, (a.z * b.x - a.x * b.z) * scaling, (a.x*b.y - a.y*b.x) * scaling};
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

void kernelLauncher(uchar4 *d_out, float cam_distance,  float cameratheta, float cameraphi, int width, int height,float scaling, int count, float *spheres_coords, float* spheres_rads){

    float3 up = {0.0f ,0.0f, 1.0f};
    float sinphi = sinf(cameraphi);
    float cosphi = cosf(cameraphi);
    float sintheta = sinf(cameratheta);
    float costheta = cosf(cameratheta);
    printf("Camera: %f %f\n", cameratheta, cameraphi);
    // printf("sinphi: %f, cosphi: %f, sintheta: %f, costheta: %f\n", sinphi, cosphi, sintheta, costheta);


    float3 r = {costheta*sinphi, sintheta*sinphi, cosphi};
    printf(
        "Camera Direction: %f %f %f\n",
        r.x, r.y, r.z
    );
    float3 plane_right = normalize_vector(cross_product(up, r, (cameraphi<0.0f)? -1.0f : 1.0f));
    // printf(r.x, r.y, r.z);
    printf(
        "Camera: %f %f %f\n",
        r.x, r.y, r.z
    );
    printf(
        "Plane Right: %f %f %f\n",
        plane_right.x, plane_right.y, plane_right.z
    );
    float3 plane_up = normalize_vector(cross_product(r, plane_right));
    printf(
        "Plane Up: %f %f %f\n\n",
        plane_up.x, plane_up.y, plane_up.z
    );
    // if(r.z < 0.0f){
    //     plane_up.x *= -1.0f;
    //     plane_up.y *= -1.0f;
    //     // plane_up.z *= -1.0f;
    // }
    float3 plane_normal = {-r.x, -r.y, -r.z};

    const dim3 blockSize(TX, TY);
    const dim3 gridSize((width + TX - 1) / TX, (height + TY - 1) / TY);
    

    ray_tracing<<<gridSize, blockSize>>>(d_out, scaling, plane_right, plane_up, plane_normal, cam_distance, width, height);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();

    


    return;
}
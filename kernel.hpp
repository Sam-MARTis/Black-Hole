#pragma once

struct uchar4;
// struct float2;
// struct float3;

void kernelLauncher(uchar4 *d_out, float camerax, float cameray, float cam_distance,  int width, int height,float scaling, int count, float *spheres_coords, float* spheres_rads);
void testKernelLauncher(uchar4 *d_out, int width, int height);
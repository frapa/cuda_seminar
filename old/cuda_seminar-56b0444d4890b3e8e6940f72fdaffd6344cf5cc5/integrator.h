#ifndef INTEGRATOR
#define INTEGRATOR

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void stepSimulation(float *T, float K, float2 *vertices);
__global__ void stepSimulation2D(float *T, float *K, float *dT, unsigned n_loop, uchar4 *tex);

#ifdef __cplusplus
}
#endif

#endif

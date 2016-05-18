#ifndef INTEGRATOR
#define INTEGRATOR

#ifdef __cplusplus
extern "C" {
#endif

__global__ void stepSimulation(float *T, float K, float2 *vertices);

#ifdef __cplusplus
}
#endif

#endif

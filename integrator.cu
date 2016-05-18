#include "integrator.h"

__device__ void integrate(unsigned lid, float K)
{
	extern __shared__ float local_T[];

	local_T[lid+1] += K * (local_T[lid+2] + local_T[lid] - 2*local_T[lid+1]);
}

__device__ void loadSharedMemory(unsigned gid, unsigned lid, float *T)
{
	extern __shared__ float local_T[];

	local_T[lid+1] = T[gid+1];

	if (lid == 0) {
		local_T[0] = T[gid];
		local_T[blockDim.x + 1] = T[gid + blockDim.x + 1];
	}
}

__device__ void copyData(unsigned gid, unsigned lid, float *T, float2 *vertices)
{
	extern __shared__ float local_T[];
	float len = gridDim.x * blockDim.x / 2;

	T[gid+1] = local_T[lid+1];

	vertices[gid].x = (gid - len) / len;
	vertices[gid].y = local_T[lid+1] - 1.0f;
}

__global__ void stepSimulation(float *T, float K, float2 *vertices) {
	unsigned lid = threadIdx.x;
	unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ float local_T[];

	loadSharedMemory(gid, lid, T);
	__syncthreads();

	integrate(lid, K);

	copyData(gid, lid, T, vertices);
	__syncthreads();
}

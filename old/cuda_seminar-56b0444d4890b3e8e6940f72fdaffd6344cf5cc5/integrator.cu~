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

// blockDim.x * gridDim.x should be exacly the length of T - 2 (2 for the boundaries)
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

// ---- 2D ----------------------------------------------------------------------------

typedef struct {
	int2 gid;
	unsigned lid_1d, gid_1d;
	unsigned lid_1d_nb;
	unsigned lw, gw, n_loop;
	float *local_T;
} UsefulConstants;

__device__ void integrate2D(const UsefulConstants consts, float *T, float *K, float *dT)
{
	// Same boring thing to save typing
	unsigned lw = consts.lw; 
	float *local_T = consts.local_T;
	unsigned lid_1d = consts.lid_1d;
	unsigned gid_1d = consts.gid_1d;

	// convolve local_T with the laplacian operator:
	//
	//     0  1  0
	//     1 -4  1
	//     0  1  0
	//
	// and save the result in local_result
	unsigned i;
	for (i = 0; i < consts.n_loop; ++i) {
		T[gid_1d+i] += K[gid_1d+i] *
			(local_T[lid_1d+1+i] + local_T[lid_1d-1+i] + local_T[lid_1d+lw+i] + local_T[lid_1d-lw+i] - 4*local_T[lid_1d+i])
			+ dT[gid_1d+i]; // + dT used to increment the temperature at the heater, for example
	}
}

// Copies to shared memory the temperature in each point for faster access.
// Does some fancy stuff to fill the boundaries.
__device__ void loadSharedMemory2D(const UsefulConstants consts, float *T)
{
	// for economy of chars
	unsigned gw = consts.gw; 
	unsigned lw = consts.lw; 
	float *local_T = consts.local_T;
	unsigned lid_1d = consts.lid_1d;
	unsigned gid_1d = consts.gid_1d;
	unsigned n_loop = consts.n_loop;

	// each thread copies n_loop pixels in a row to shared memory
	unsigned i;
	for (i = 0; i < n_loop; ++i) {
		local_T[lid_1d + i] = T[gid_1d + i];
	}

	// We have lots of pixels to be copied at the border...
	// We assume here that the block is a square

	// Number of border pixels copied by each thread:
	unsigned c = 4 * n_loop / blockDim.y;
	// scalar index of the thread inside a block, starting from 0
	unsigned k = threadIdx.x + blockDim.x * threadIdx.y;
	// Number of thread needed to copy each of the four borders
	// (should be 1/4 of the total number of threads)
	unsigned thread_per_border = blockDim.y / c;
	// Thread number along border, starting with 0
	unsigned k_along_border = k % thread_per_border;
	// Starting offset along border
	unsigned boffset = c * k_along_border;
	// Number of the border for current thread,
	// from 0 to 3 in the following order: top, left, right, bottom
	unsigned bnum = k / (thread_per_border);

	// g stands for global
	unsigned offset, mul, goffset, gmul;
	// This fills the previous variables according to bnum
	// used to move along border afterwards
	if (bnum == 0) {
		offset = goffset = 1;
		mul = gmul = 1;
	} else if (bnum == 1) {
		offset = lw;
		goffset = gw;
		mul = lw;
		gmul = gw;
	} else if (bnum == 2) {
		offset = 2*lw - 1;
		goffset = 2*gw - 1;
		mul = lw;
		gmul = gw;
	} else if (bnum == 3) {
		offset = lw * (blockDim.y + 1) + 1;
		goffset = gw * (blockDim.y + 1) + 1;
		mul = gmul = 1;
	}

	// fill borders
	unsigned j, pos_along_border;
	// global index of the first pixel in block, border included
	unsigned gid_1d_start = blockIdx.y * blockDim.y * gw + blockIdx.x + blockDim.x * n_loop;
	for (j = 0; j < c; ++j) {
		pos_along_border = j + boffset;

		local_T[offset + pos_along_border*mul] =
			T[gid_1d_start + goffset + pos_along_border*gmul];
	}
}


__device__ void drawToTexture(const UsefulConstants consts, float * T, uchar4 *tex)
{
	// Did I already say it's just a repetition?
	int2 gid = consts.gid;
	unsigned gw = consts.gw - 2; 
	unsigned gid_1d = consts.gid_1d;
	
	unsigned i, idx;
	for (i = 0; i < consts.n_loop; ++i) {
		idx = gid.y * gw + gid.x + i;
		tex[idx].x = T[gid_1d + i];
		tex[idx].y = T[gid_1d + i];
		tex[idx].z = T[gid_1d + i];
		tex[idx].w = 255;
	}
}

/*
	T -> Initial array with temperature at each point, including borders (= boundary conditions);
	K -> Array specifying the thermal conductivity at each point. Without borders;
	dT -> Array specifying the increase in temperature at each step for each point, without borders;
	n_loop -> How many pixels in a row should each thread compute. Must be exact fraction of blockDim.x;
	tex -> Handle to texture shared with OpenGl to display the result;
 */

// blockDim.x * blockDim.y * gridDim.x * gridDim.y * n_loop should be exacly the length of
// T - 2 * (gridDim.x * blockDim.x + gridDim.y * blockDim.y) - 4 (subtraction of something for the boundaries)
__global__ void stepSimulation2D(float *T, float *K, float *dT, unsigned n_loop, uchar4 *tex) {
	// Calculate some constants useful around
	int2 gid;
	gid.x = (blockIdx.x * blockDim.x + threadIdx.x) * n_loop;
	gid.y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned gw = gridDim.x * blockDim.x * n_loop + 2; 
	unsigned lw = blockDim.x * n_loop + 2; 

	unsigned lid_1d = (threadIdx.y + 1) * lw + 1 + threadIdx.x;
	unsigned gid_1d = (gid.y + 1) * gw + 1 + gid.x;
	// local id without borders
	unsigned lid_1d_nb = threadIdx.y * lw + threadIdx.x;

    	// declare dynamically allocated shared memory
	extern __shared__ float local_T[];
	
	// create a struct to pass around all our nice constants
	UsefulConstants consts = {
		.gid = gid,
		.lid_1d = lid_1d,
		.gid_1d = gid_1d,
		.lid_1d_nb = lid_1d_nb,
		.gw = gw,
		.lw = lw,
		.n_loop = n_loop,
		.local_T = local_T
	};

	// load data to shared memory
	loadSharedMemory2D(consts, T);
	__syncthreads();

	// carry out the integration
	integrate2D(consts, T, K, dT);

	// compy data back to global memory and fill the texture for visualization with OpenGL
	drawToTexture(consts, T, tex);
	__syncthreads();
}

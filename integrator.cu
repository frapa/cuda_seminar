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
	unsigned lid_1d_nb, gid_1d_nb;
	unsigned lw, gw, n_loop;
	float *local_T;
} UsefulConstants;


/*******************************************************************************
 * INTEGRATE 2D
 *******************************************************************************/
__device__ void integrate2D(const UsefulConstants consts, float *T, float *K, float *dT, int *d_operation)
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
	// come mai non usiamo local_T e poi salviamo tutto indietro? 
	unsigned i;
	for (i = 0; i < consts.n_loop; ++i) {
		T[gid_1d+i] += K[gid_1d+i-1-consts.gw] *
			  (local_T[lid_1d+1+i] + local_T[lid_1d-1+i] + local_T[lid_1d+lw+i] 
			   + local_T[lid_1d-lw+i] - 4*local_T[lid_1d+i])
			+ dT[gid_1d+i-1-consts.gw]; // + dT used to increment the temperature at the heater, for example
		//d_operation[gid_1d+i] = 255;
		
		//T[gid_1d+i] = local_T[lid_1d+i]; // To test loadSharedMemory2D
	}
}


/*******************************************************************************
 * LOAD SHARED MEMORY 2D
 *******************************************************************************
 * Copies to shared memory the temperature in each point for faster access.
 * Does some fancy stuff to fill the boundaries.
 */
__device__ void loadSharedMemory2D(const UsefulConstants consts, float *T, int *d_operation)
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
		d_operation[gid_1d+i] = 255;
	}
	
	/* assume the block is a square:
	 * # border pixels = blockDim.y * 2 + blockDim.x * n_loop * 2
	 * # border pixels = blockDim.y * 4
	 * 
	 * k:		scalar index of the thread inside a block
	 * bnum:	side number associated with the core, 
	 * 		from 0 to 3 in the following order: top, left, right, 
	 *		bottom. if>4 thread is ignored
	 * c:		number of core to be copied by each thread
	 * 		= 4 * blockDim.x * n_loop / (blockDim.y * blockDim.x)
	 * 		remember blockDim.x*n_loop = blockDim.y
	 * 		usually < 1 -> c = 0, must add 1.
	 * offset, mul, goffset, gmul:
	 * 		g stands for global. offset = starting point,
	 * 		mul = step of iteration (allow you to go down a column)
	 * gid_1d_start:
	 * 		global index of the first pixel in block, 
	 * 		border included
	 */
	unsigned k = threadIdx.x + blockDim.x * threadIdx.y;
	unsigned c = 4 / blockDim.x + 1;
	unsigned bnum = k*c / blockDim.y;
	
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
		offset = lw * (blockDim.y + 1) + 2;
		goffset = gw * (blockDim.y + 1) + 2;
		mul = gmul = 1;
	}

	// fill borders
	unsigned j;
	/* ho tolto il + 1 perchè la numerazione dovrebbe andare da 0 a
	 * blockDim.x*n_loop, non da 0 a (blockDim.x + 1) * n_loop 
	 */
	unsigned gid_1d_start = blockIdx.y * blockDim.y * gw
				+ blockIdx.x * blockDim.x * n_loop - 1;
	for (j = 0; j < c; ++j) {
		local_T[offset + j*mul] = T[gid_1d_start + goffset + j*gmul];
		d_operation[gid_1d_start + goffset + j*gmul] = 100;
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


/*******************************************************************************
 * STEP SIMULATION 2D
 *******************************************************************************
 * T:	    Initial array with temperature at each point, including borders 
 * 	    (= boundary conditions);
 * K:	    Array specifying the thermal conductivity at each point. 
 * 	    Without borders;
 * dT:	    Array specifying the increase in temperature at each step 
 * 	    for each point, without borders;
 * n_loop:  How many pixels in a row should each thread compute. 
 * 	    must be exact fraction of blockDim.x;
 * tex:	    Handle to texture shared with OpenGl to display the result;
 *
 * blockDim.x * blockDim.y * gridDim.x * gridDim.y * n_loop should be exacly the length of
 * T - 2 * (gridDim.x * blockDim.x + gridDim.y * blockDim.y) - 4 (subtraction of something for the boundaries)
 */
__global__ void stepSimulation2D(float *T, float *K, float *dT, unsigned n_loop, 
				 uchar4 *tex, int *d_operation) 
{
	// Calculate some constants useful around
	int2 gid;	// saves for each thread gives the starting T[i,j] not considering borders
	gid.x = (blockIdx.x * blockDim.x + threadIdx.x)*n_loop;
	gid.y = blockIdx.y * blockDim.y + threadIdx.y;
	/*
	gid.x = 1 + (blockIdx.x * blockDim.x + threadIdx.x) * n_loop;
	gid.y = 1 + blockIdx.y * blockDim.y + threadIdx.y;
	*/
		
	unsigned gw = gridDim.x * blockDim.x * n_loop + 2; // global width for grid
	unsigned lw = blockDim.x * n_loop + 2; // local width for block
	
	// local and global id in 1dim for T[i,j]
	// [MODIFICA] aggiunto "* n_loop" in entrambe le righe
	unsigned lid_1d = (threadIdx.y + 1) * lw + 1 + threadIdx.x * n_loop;
	unsigned gid_1d = (gid.y + 1) * gw + 1 + gid.x;
	// local id without borders
	unsigned lid_1d_nb = threadIdx.y * lw + threadIdx.x * n_loop;
	unsigned gid_1d_nb = gid.y * gw + gid.x;

    	// declare dynamically allocated shared memory
	extern __shared__ float local_T[];
	
	// create a struct to pass around all our nice constants
	UsefulConstants consts = {
		.gid = gid,
		.lid_1d = lid_1d,
		.gid_1d = gid_1d,
		.lid_1d_nb = lid_1d_nb,
		.gid_1d_nb = gid_1d_nb,
		.gw = gw,
		.lw = lw,
		.n_loop = n_loop,
		.local_T = local_T
	};

	// load data to shared memory
	loadSharedMemory2D(consts, T, d_operation);
	__syncthreads();

	// carry out the integration
	integrate2D(consts, T, K, dT, d_operation);
	// copy data back to global memory and fill the texture for visualization with OpenGL
	drawToTexture(consts, T, tex);
	__syncthreads();
}

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
__device__ void integrate2D(const UsefulConstants consts, float *T, float *K, float *dT)
{
	// Same boring thing to save typing
	unsigned lw = consts.lw; 
	float *local_T = consts.local_T;
	unsigned lid_1d = consts.lid_1d;
	unsigned gid_1d = consts.gid_1d;	
	unsigned gid_1d_nb = consts.gid_1d_nb;
	//unsigned gid_1d_nb = consts.gid_1d_nb;

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
		T[gid_1d+i] += K[gid_1d_nb+i] *
			  (local_T[lid_1d+1+i] + local_T[lid_1d-1+i] + local_T[lid_1d+lw+i] 
			   + local_T[lid_1d-lw+i] - 4*local_T[lid_1d+i])
			+ dT[gid_1d_nb+i]; // + dT used to increment the temperature at the heater, for example
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
		//d_operation[gid_1d+i] = 255;
	}
	__syncthreads();
	
	/* Fill the borders with more threads:
	 * assume the block is a square:
	 * # border pixels = blockDim.y * 2 + blockDim.x * n_loop * 2
	 * # border pixels = blockDim.y * 4
	 * 
	 * k:		scalar index of the thread inside a block
	 * bnum:	side number associated with the core, 
	 * 		from 0 to 3 in the following order: top, left, right, 
	 *		bottom. if>4 thread is ignored
	 * c:		number of pixels to be copied by each thread
	 * 		= 4 * blockDim.x * n_loop / (blockDim.y * blockDim.x)
	 * 		remember blockDim.x*n_loop = blockDim.y
	 * 		usually < 1 -> c = 0, must add 1.
	 * offset, mul, goffset, gmul:
	 * 		g stands for global. offset = starting point,
	 * 		mul = step of iteration (allow you to go down a column)
	 * gid_1d_start:
	 * 		global index of the first pixel in block, 
	 * 		border included
	 * side_id:	thread id on the side, considering 0 the first element of the
	 *			side
	 */
	//unsigned c = 4 * n_loop / blockDim.y + 1;
	//unsigned k = threadIdx.x + blockDim.x * threadIdx.y;

	unsigned k = threadIdx.x + blockDim.x * threadIdx.y;
	unsigned c = 4 / blockDim.x;
	c = c < 1 ? 1 : c; 
	unsigned bnum = k*c / blockDim.y;
	unsigned gid_1d_start = blockIdx.y * blockDim.y * gw
				+ blockIdx.x * blockDim.x * n_loop;
	unsigned side_id = c*k - bnum * blockDim.y;
	unsigned offset, mul, goffset, gmul;

	if (bnum == 0) {
		offset = goffset = 1;
		mul = gmul = 1;
		// fill borders
		unsigned j;
		for (j = 0; j < c; ++j) {
			local_T[offset + (j+side_id)*mul] = T[gid_1d_start + goffset + (j+side_id)*gmul];
		}
	} else if (bnum == 1) {
		offset = lw;
		goffset = gw;
		mul = lw;
		gmul = gw;
		// fill borders
		unsigned j;
		for (j = 0; j < c; ++j) {
			local_T[offset + (j+side_id)*mul] = T[gid_1d_start + goffset + (j+side_id)*gmul];
		}
	} else if (bnum == 2) {
		offset = 2*lw - 1;
		goffset = gw + lw -1;
		mul = lw;
		gmul = gw;
		// fill borders
		unsigned j;
		for (j = 0; j < c; ++j) {
			local_T[offset + (j+side_id)*mul] = T[gid_1d_start + goffset + (j+side_id)*gmul];
		}
	} else if (bnum == 3) {
		offset = lw * (blockDim.y + 1) + 1;
		goffset = gw * (blockDim.y + 1) + 1;
		mul = gmul = 1;
		// fill borders
		unsigned j;
		for (j = 0; j < c; ++j) {
			local_T[offset + (j+side_id)*mul] = T[gid_1d_start + goffset + (j+side_id)*gmul];
		}
	}

	// Fill the borders with just one thread, variables as before
	/*unsigned k = threadIdx.x + blockDim.x * threadIdx.y;
	unsigned gid_1d_start = blockIdx.y * blockDim.y * gw
				+ blockIdx.x * blockDim.x * n_loop;
	unsigned offset, mul, goffset, gmul;
	if (k == 0) {
		unsigned j;

		// fill border 0, works fine
		offset = goffset = 1;
		mul = gmul = 1;
		for (j = 0; j < blockDim.y; ++j) {
			local_T[offset + j*mul] = T[gid_1d_start + goffset + j*gmul];
			//d_operation[gid_1d_start + goffset + j*gmul] = 100;
		}
		
		// fill border 1, works fine
		offset = lw;
		goffset = gw;
		mul = lw;
		gmul = gw;
		for (j = 0; j < blockDim.y; ++j) {
			local_T[offset + j*mul] = T[gid_1d_start + goffset + j*gmul];
			//d_operation[gid_1d_start + goffset + j*gmul] = 100;
		}
	
		// fill border 2
		offset = 2*lw - 1;
		goffset = gw + lw -1;
		mul = lw;
		gmul = gw;
		for (j = 0; j < blockDim.y; ++j) {
			local_T[offset + j*mul] = T[gid_1d_start + goffset + j*gmul];
			//d_operation[gid_1d_start + goffset + j*gmul] = 100;
		}
	
		// fill border 3
		offset = lw * (blockDim.y + 1) + 1;
		goffset = gw * (blockDim.y + 1) + 1;
		mul = gmul = 1;
		for (j = 0; j < blockDim.y; ++j) {
			local_T[offset + j*mul] = T[gid_1d_start + goffset + j*gmul];
			//d_operation[gid_1d_start + goffset + j*gmul] = 100;
		}	
	}*/
}


__device__ void drawToTexture(const UsefulConstants consts, float *T, float *K,
		uchar4 *tex, char show_cond)
{
	// Did I already say it's just a repetition?
	int2 gid = consts.gid;
	unsigned gw = consts.gw - 2; 
	unsigned gid_1d = consts.gid_1d;
	unsigned gid_1d_nb = consts.gid_1d_nb;
	
	unsigned i, idx;
	for (i = 0; i < consts.n_loop; ++i) {
		idx = gid.y * gw + gid.x + i;
		tex[idx].x = T[gid_1d + i];
		tex[idx].y = show_cond ? K[gid_1d_nb + i] * 1e4 : 0;
		tex[idx].z = 0;
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
 * copy_tex: Copy texture?
 */
__global__ void stepSimulation2D(float *T, float *K, float *dT, unsigned n_loop, 
				    uchar4 *tex, char copy_tex, char show_cond) 
{
	// Calculate some constants useful around
	int2 gid;	// for each thread, starting T[i,j] not considering borders
	gid.x = (blockIdx.x * blockDim.x + threadIdx.x)*n_loop;
	gid.y = blockIdx.y * blockDim.y + threadIdx.y;
		
	unsigned gw = gridDim.x * blockDim.x * n_loop + 2; // global width for grid
	unsigned lw = blockDim.x * n_loop + 2; // local width for block
	
	// local and global id in 1dim for T[i,j]
	unsigned lid_1d = (threadIdx.y + 1) * lw + 1 + threadIdx.x * n_loop;
	unsigned gid_1d = (gid.y + 1) * gw + 1 + gid.x;
	// local id without borders
	unsigned lid_1d_nb = threadIdx.y * (lw-2) + threadIdx.x * n_loop;
	unsigned gid_1d_nb = gid.y * (gw-2) + gid.x;

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
	loadSharedMemory2D(consts, T);
	__syncthreads();

	// carry out the integration
	integrate2D(consts, T, K, dT);
	
	// copy data back to global memory and fill the texture for visualization with OpenGL
	if (copy_tex) {
	    drawToTexture(consts, T, K, tex, show_cond);
	}
	
	__syncthreads();
}

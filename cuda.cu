#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "gl_helper.h"

#define REAL float
#define VECTOR float4
#define DT 0.01
#define SCALE 1000.0f

unsigned n = 1024;
unsigned thn = 256;
unsigned bn = (n / thn);
unsigned steps = 1024;
unsigned size = sizeof(VECTOR) * n;
unsigned c = 0;

REAL *pos, *vel;
REAL *dpos, *dvel;

__device__ float3 computeBodyBodyAccel(VECTOR me, VECTOR other, float3 a)
{
	float3 r;
	// mass of other object
	float m = other.w;

	r.x = other.x - me.x;
	r.y = other.y - me.y;
	r.z = other.z - me.z;

	// distance squared
	float d3 = r.x*r.x*r.x + r.y*r.y*r.y + r.z*r.z*r.z;

	if (d3 == 0) {
		return a;
	}

	float s = m / d3;

	a.x += s * r.x;
	a.y += s * r.y;
	a.z += s * r.z;

	return a;
}

__device__ float3 computeTileAccel(VECTOR me, float3 a)
{
	extern __shared__ VECTOR localPositions[];

	int i;
	for (i = 0; i < blockDim.x; i++) {
		a = computeBodyBodyAccel(me, localPositions[i], a);
	}

	return a;
}

// Leapfrog
__device__ void integrate(unsigned idx, VECTOR *pos, VECTOR *vel, float3 a)
{
	pos[idx].x += vel[idx].x * DT;
	pos[idx].y += vel[idx].y * DT;
	pos[idx].z += vel[idx].z * DT;

	vel[idx].x += a.x * DT;
	vel[idx].y += a.y * DT;
	vel[idx].z += a.z * DT;
}

__device__ void create_vertices(unsigned idx, VECTOR *vertices, VECTOR *pos)
{
	vertices[idx].x = pos[idx].x / SCALE;
	vertices[idx].y = pos[idx].y / SCALE;
	vertices[idx].z = 0;
	vertices[idx].w = 0;
}

__global__ void simulate_step(VECTOR *vertices, void *init_pos, void *init_vel, unsigned n)
{
	VECTOR *pos = (VECTOR *)init_pos;
	VECTOR *vel = (VECTOR *)init_vel;

	extern __shared__ VECTOR localPositions[];

	unsigned i, tile;
	unsigned me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	create_vertices(me_idx, vertices, pos);

	float3 a = {0.0f, 0.0f, 0.0f};
	VECTOR me = pos[me_idx];
	for (i = 0, tile = 0; i < n; i += blockDim.x, tile++) {
		unsigned idx = tile * blockDim.x + threadIdx.x;
		// copy positions to local memory
		localPositions[threadIdx.x] = pos[idx];
		__syncthreads();

		// Compute accelerations
		a = computeTileAccel(me, a);
		__syncthreads();
	}

	integrate(me_idx, pos, vel, a);
}

void display_frame()
{
	VECTOR *vertices = (VECTOR *)map_resource();
	
	simulate_step<<<bn, thn, size / bn>>>(vertices, dpos, dvel, n);
	printf("%d\n", c++);
	sleep(5);

	unmap_and_draw();
}

void setup()
{
	pos = (REAL *) malloc(size);
	vel = (REAL *) malloc(size);

	cudaMalloc(&dpos, size);
	cudaMalloc(&dvel, size);

	// Fill with random data
	srand(time(NULL));
	unsigned i;
	for (i = 0; i < n; i += 4) {
		pos[i] = (double)(2 * rand() - RAND_MAX) / RAND_MAX * SCALE;
		pos[i+1] = (double)(2 * rand() - RAND_MAX) / RAND_MAX * SCALE;
		pos[i+2] = 0;
		pos[i+3] = 0;

		vel[i] = 0;
		vel[i+1] = 0;
		vel[i+2] = 0;
		vel[i+3] = 0;
	}

	cudaMemcpy(dpos, pos, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dvel, vel, size, cudaMemcpyHostToDevice);
	
	register_array(size, n);
}

void cleanup()
{
	free(pos);
	free(vel);
	cudaFree(dpos);
	cudaFree(dvel);

	freeGL();
}

int main(int argc, char **argv)
{
	initGL(&argc, argv, "Solar system simulation", display_frame);
	
	setup();

	glutMainLoop();

	cleanup();

	return 0;
}

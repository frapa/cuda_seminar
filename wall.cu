#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "integrator.h"
#include "gl_helper.h"

unsigned n, thread_num;
size_t size_shared;
float *T_device;
float K;

void step()
{
	float2 *vertices = (float2 *) map_resource();

	stepSimulation<<<n/thread_num, thread_num, size_shared>>>(T_device, K, vertices);

	unmap_and_draw();
}

int main(int argc, char **argv)
{
	initGL(&argc, argv, "Heat equation", step);
	
	n = 2304;
	thread_num = 192;
	K = 0.000005 * 0.01 * n*n;
	size_t size = sizeof(float) * (n + 2);
	size_shared = sizeof(float) * (thread_num + 2);

	printf("%f\n", K);

	float *T = (float *)malloc(size);
	float *T2 = (float *)malloc(size);
	cudaMalloc(&T_device, size);
	
	register_array(n * 2, sizeof(float), n);

	T[0] = 0;
	T[n+1] = 2;
	unsigned i;
	for (i = 1; i <= n; ++i) {
		T[i] = 1;
	}

	cudaMemcpy(T_device, T, size, cudaMemcpyHostToDevice);

	glutMainLoop();

	//stepSimulation<<<n/thread_num, thread_num, size_shared>>>(T_device, K);

	/*cudaMemcpy(T2, T_device, size, cudaMemcpyDeviceToHost);

	FILE *f;
	f = fopen("out.txt", "w");
	for (i = 0; i <= n+1; ++i) {
		fprintf(f, "%f %f %f\n", (float)i / n, T[i], T2[i]);
	}
	fclose(f);*/

	free(T);
	free(T2);
	cudaFree(T_device);
        
    return 0;
}

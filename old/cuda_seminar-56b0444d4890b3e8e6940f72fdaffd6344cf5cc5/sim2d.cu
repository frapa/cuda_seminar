#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "tiffio.h"

#include "integrator.h"
#include "gl_helper.h"

#define DEBUG

// global constants
unsigned w, h, n_loop, loop_done, watch;
float heating_level = 0;
dim3 block_num, thread_num;
size_t size_shared;
float *dT;
float *T_device, *K_device, *dT_device, *tmp;
uchar4 *image;
double cpu_time, cpu_step;
size_t temp_size;

void readTiff(char *filename, float **raster, unsigned *w, unsigned *h, 
	      float scale)
{
	// Open file
	TIFF* img = TIFFOpen(filename, "r");
	
	// read file size
	TIFFGetField(img, TIFFTAG_IMAGEWIDTH, w);
	TIFFGetField(img, TIFFTAG_IMAGELENGTH, h);

	// allocate memory
	uint32 *tmp_raster = (uint32*) _TIFFmalloc((*w)*(*h) * sizeof(uint32));
	*raster = (float *) malloc((*w)*(*h) * sizeof(float));

	if (tmp_raster != NULL) {
		// finally read image
	    if (TIFFReadRGBAImage(img, *w, *h, tmp_raster, 0)) {
			// separate red channnel (we have grayscale images so it's the same)
			// and convert to float with scale
			unsigned i;
			for (i = 0; i < (*w)*(*h); ++i) {
				(*raster)[i] = ((float)TIFFGetR(tmp_raster[i])) * scale;
			}
	    }
	    _TIFFfree(tmp_raster);
	} else {
		printf("ERROR: cannot read file '%s'", filename);
	}

	TIFFClose(img);
}

void interpolate_array(float *in, float *out, unsigned size, float opacity)
{
    unsigned i;
    for (i = 0; i < size; ++i)
        out[i] = in[i] * opacity;
}

void on_key_pressed(unsigned char key, int x, int y)
{
    switch(key) {
        case '+':
            if (heating_level < 1) {
                unsigned size = w * h;
                heating_level += 0.1;
                interpolate_array(dT, tmp, size, heating_level);
                cudaMemcpy(dT_device, dT, size * sizeof(float), cudaMemcpyHostToDevice);
            }
            break;
        case '-':
            if (heating_level > 0) {
                unsigned size = w * h;
                heating_level -= 0.1;
                interpolate_array(dT, tmp, size, heating_level);
                cudaMemcpy(dT_device, dT, size * sizeof(float), cudaMemcpyHostToDevice);
            }
            break;
    }
}

void step()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	cudaArray *tex = map_texture();
    clock_t start_host, end_host; // Used to check time of execution
	int i;

    start_host = clock();
	stepSimulation2D<<<block_num, thread_num, size_shared>>>
	    (T_device, K_device, dT_device, n_loop, image);
	cudaError_t error = cudaDeviceSynchronize();
	end_host=clock();

	if (error != cudaSuccess) {
		printf("Error while running kernel: %s\n", cudaGetErrorString(error));
	}
	
	// Copy data for controlling the correct execution of the simulation
	float *T_check;
	T_check = (float*)malloc(temp_size);
	
	if (loop_done == watch){
	  cudaMemcpy(T_check, T_device, temp_size, cudaMemcpyDeviceToHost);	
	  
	  FILE *ftemp ;
	  ftemp = fopen("check/temperature.txt", "w");
	  
	  for (i=512*256; i<512*256+512; i++){
	      fprintf(ftemp, "%f ", T_check[i]);
	      fprintf(ftemp, "\n\n\n");
	  }
	  for (i=0; i<512; i++){
	      fprintf(ftemp, "%f ", T_check[256+i*512]);
	  }
	}
	
	cpu_step = ((double)  (end_host - start_host));
	cpu_time += cpu_step / CLOCKS_PER_SEC;
	loop_done += 1;

	cudaMemcpyToArray(tex, 0, 0, image, w*h*4, cudaMemcpyDeviceToDevice);

	unmap_and_draw();
	
	glutSwapBuffers();
	glutPostRedisplay();
}

int main(int argc, char **argv)
{
	// First check if a directory was given
	if (argc < 2) {
	    printf("No simulation given.\n");
		return -1;
	}

	// Build filenames
	char *simulation_folder = argv[1];
	unsigned len = strlen(simulation_folder);
	char *temperature = (char *)malloc(len + 17);
	char *conductivity = (char *)malloc(len + 18);
	char *heating = (char *)malloc(len + 13);

	strcpy(temperature, simulation_folder);
	strcpy(temperature + len, "temperature.tiff");
	strcpy(conductivity, simulation_folder);
	strcpy(conductivity + len, "conductivity.tiff");
	strcpy(heating, simulation_folder);
	strcpy(heating + len, "heating.tiff");

	// read files
	float *T, *K, *dT;
	readTiff(temperature, &T, &w, &h, 1);
	readTiff(conductivity, &K, &w, &h, 0.0001);
	readTiff(heating, &dT, &w, &h, 1);
	printf("Simulation size: %ux%u\n", w, h);
	
	// Setup other interesting stuff
	// Parse other command line arguments
	unsigned square_side = 64;
	n_loop = 8;
	if (argc > 2) {
	    unsigned j;
	    for (j = 2; j < argc; ++j) {
	        if (!strcmp(argv[j], "-bs")) {
		        sscanf(argv[j+1], "%u", &square_side);
	        } else if (!strcmp(argv[j], "-n")) {
		        sscanf(argv[j+1], "%u", &n_loop);
	        } else if (!strcmp(argv[j], "-l")) {
		        sscanf(argv[j+1], "%u", &watch);
	        }
	    }
	}
	printf("-------------------------\n");
	printf("Loops per thread: %u\n", n_loop);
	printf("Block size: %ux%u (%u threads per block)\n",
	    square_side, square_side/n_loop, square_side*square_side/n_loop);
	
	// for heating
	size_t param_size = w * h * sizeof(float);
	temp_size = (w + 2) * (h + 2) * sizeof(float);
	tmp = (float *) malloc(param_size);
	
	// dimensions of grid, blocks and shared memory
	thread_num.x = square_side / n_loop;
	thread_num.y = square_side;
	block_num.x = w / square_side;
	block_num.y = h / square_side;
	size_shared = sizeof(float) * (square_side + 2) * (square_side + 2);
	
	printf("Grid size: %ux%u\n", block_num.x, block_num.y);
	printf("Shared memory: %.2f Kb\n", size_shared / 1024.f);
	
	// Copy input to device
	cudaMalloc(&T_device, temp_size);
	cudaMemcpy(T_device, T, temp_size, cudaMemcpyHostToDevice);
	
	cudaMalloc(&K_device, param_size);
	cudaMemcpy(K_device, K, param_size, cudaMemcpyHostToDevice);
	
	cudaMalloc(&dT_device, param_size);
	cudaMemcpy(dT_device, dT, param_size, cudaMemcpyHostToDevice);

	cudaMalloc(&image, w*h*4);

	// Now that we are done loading the simulation, we start OpenGL
	initGL(&argc, argv, "Heat equation", step);

	cudaSetDevice(0);
	cudaGLSetGLDevice(0);

	register_texture(w, h);
	glutKeyboardFunc(on_key_pressed);
	
	//register_array(n * 2, sizeof(float), n);

	// Start simulation
	loop_done = 0;
	cpu_time = 0;
	glutMainLoop();
	
	// Print time statistics
	printf("Total Time: %f\n", cpu_time);
	printf("Mean Time per Step: %f\n", cpu_time/(double)loop_done);
	
	// cleanup
	free(T);
	free(K);
	free(dT);
	free(tmp);
	cudaFree(T_device);
        
    return 0;
}

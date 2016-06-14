#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "tiffio.h"

#include "integrator.h"

#define DEBUG

// global constants
/* watch saves the step number to be controlled 
 */
unsigned w, h, n_loop, loop_done, watch;
float heating_level = 0;
dim3 block_num, thread_num;
size_t size_shared;
float *dT;
float *T_device, *K_device, *dT_device, *tmp;
int *operation, *d_operation;	// Serve a controllare che il programma ricalcoli tutti i punti
uchar4 *image;
double cpu_time, cpu_step;
size_t temp_size, op_size;


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

void step()
{
	clock_t start_host, end_host; // Used to check time of execution
	
	while(1){
		// START SIMULATION
		start_host = clock();
		stepSimulation2D<<<block_num, thread_num, size_shared>>>
		    (T_device, K_device, dT_device, n_loop, image, d_operation);
		cudaError_t error = cudaDeviceSynchronize();
		end_host=clock();

		if (error != cudaSuccess) {
			printf("Error while running kernel: %s\n", cudaGetErrorString(error));
		}
	
		cpu_step = ((double)  (end_host - start_host));
		cpu_time += cpu_step / CLOCKS_PER_SEC;
		loop_done += 1;

		// Print time statistics
		FILE *ftime;
		ftime = fopen("check/mean_time.txt", "w");
		if (ftime == NULL){
		  	printf("\nError while opening file mean_time.txt\n");
		  	perror("Error while opnening file mean_time.txt");
		  	exit(1);
		}
		fprintf(ftime, "Total Time: %f\nMean Time per Step: %f", cpu_time, 
				cpu_time/(double)loop_done);
		fclose(ftime);
	
		ftime = fopen("check/exe_time.txt", "a");
		if (ftime == NULL){
		  	printf("\nError while opening file mean_time.txt\n");
		  	perror("Error while opnening file mean_time.txt");
		  	exit(1);
		}
		fprintf(ftime, "%f\n", cpu_step / CLOCKS_PER_SEC);
		fclose(ftime);
	}
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
	readTiff(conductivity, &K, &w, &h, 0.001);	
	// 0.01 is unstable, 0.001 is quite stable 
	readTiff(heating, &dT, &w, &h, 0.0001);
	// if scale factor too high temperature overflow
	printf("Simulation size: %ux%u\n", w, h);
	
	// check input
	/*int i;
	FILE *ftemp ;
	ftemp = fopen("check/initial.txt", "w");
	if (ftemp == NULL){
	  	printf("\nError while opening file initial.txt\n");
	  	perror("Error while opnening file initial.txt");
	  	exit(1);
	}
	for (i=w*w/2; i<w*w/2+w; i++){
	    fprintf(ftemp, "%f\n", dT[i]);
	}
	fprintf(ftemp, "\n\n\n");
	for (i=0; i<w; i++){
	    fprintf(ftemp, "%f\n", dT[w/2 + i*w]);
	}
	
	int j;
	FILE *fgrid;
	fgrid = fopen("check/T_initial.txt", "w");
	if (ftemp == NULL){
	  	printf("\nError while opening file T_initial.txt\n");
	  	perror("Error while opnening file T_initial.txt");
	  	exit(1);
	}
	for (i=0; i<w; i++){
	  	for (j=0; j<w; j++){
	    	fprintf(fgrid, "%f ", dT[i*w + j]);
	  	}
		if (i!=w-1)
 	  	fprintf(fgrid, "\n");
	}
	fclose(fgrid);
	fclose(ftemp);*/
	
	
	// Setup other interesting stuff and parse other command line arguments
	unsigned block_side = 8;
	n_loop = 8;
	if (argc > 2) {
	    unsigned j;
	    for (j = 2; j < argc; ++j) {
	        if (!strcmp(argv[j], "-bn")) {
		        sscanf(argv[j+1], "%u", &block_side);
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
	    w/(block_side), w/(block_side*n_loop), w*w / (block_side*block_side*n_loop));
	
	// for heating
	size_t param_size = w * h * sizeof(float);
	temp_size = (w + 2) * (h + 2) * sizeof(float);
	op_size = (w + 2) * (h + 2) * sizeof(int);
	tmp = (float *) malloc(param_size);
	
	/*operation = (int*)malloc(op_size);
	for (i=0; i<(w + 2) * (h + 2); i++){
		operation[i]=255;
	}*/
	
	// dimensions of grid, blocks and shared memory
	thread_num.x = w/(block_side*n_loop);
	thread_num.y = w/(block_side);
	block_num.x = block_side;
	block_num.y = block_side;
	size_shared = sizeof(float) * (w/(block_side) + 2) * (w/(block_side) + 2);
	
	printf("Grid size: %ux%u\n", block_num.x, block_num.y);
	// Kilobit (Kb)? non Kilobyte (KB)? 
	printf("Shared memory: %.2f Kb\n", size_shared / 1024.f);
	
	// Copy input to device 
	cudaMalloc(&T_device, temp_size);
	cudaMemcpy(T_device, T, temp_size, cudaMemcpyHostToDevice);
	
	cudaMalloc(&K_device, param_size);
	cudaMemcpy(K_device, K, param_size, cudaMemcpyHostToDevice);
	
	cudaMalloc(&dT_device, param_size);
	cudaMemcpy(dT_device, dT, param_size, cudaMemcpyHostToDevice);

	cudaMalloc(&d_operation, op_size);
	cudaMemcpy(d_operation, operation, op_size, cudaMemcpyHostToDevice);

	cudaMalloc(&image, w*h*4);

	// Start simulation
	loop_done = 0;
	cpu_time = 0;
	
	

	// Looks like code after glutMainLoop(); doesn´t work... 	

	// cleanup
	free(T);
	free(K);
	free(dT);
	free(tmp);
	cudaFree(T_device);
        
    return 0;
}

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "tiffio.h"

#define DEBUG

// global constants
/* watch saves the step number to be controlled 
 */
unsigned w, h, n_loop, loop_done, watch;
float *dT;
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
 */
void integrate2D(unsigned w, unsigned h, float *T, float *K, float *dT)
{
	// convolve local_T with the laplacian operator:
	//
	//     0  1  0
	//     1 -4  1
	//     0  1  0
	//
	// and save the result in local_result
	// come mai non usiamo local_T e poi salviamo tutto indietro? 
	unsigned i, j;
	for (i = 1; i < h-1 ; ++i) {
		for (j = 1; j < w-1; ++j){
			T[i*w+j] += K[(i-1)*(w-2)+(j-1)] 
						* (T[i*w+j+1] + T[i*w+j+1] + T[(i+1)*w+j] + T[(i-1)*w+j] 
							- 4*T[i*w+j])
						+ dT[(i-1)*(w-2)+(j-1)]; 
		}
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
	
	// for heating
	size_t param_size = w * h * sizeof(float);
	temp_size = (w + 2) * (h + 2) * sizeof(float);
	op_size = (w + 2) * (h + 2) * sizeof(int);
	
	// Start simulation
	loop_done = 0;
	cpu_time = 0;
	clock_t start_host, end_host; // Used to check time of execution
	
	while(1){	
		// Simulation
		start_host = clock();
		integrate2D(w, h, T, K, dT);
		end_host=clock();
		
		// Evaluation of exe time
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

	// cleanup
	free(T);
	free(K);
	free(dT);
       
    return 0;
}

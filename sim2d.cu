#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "tiffio.h"

#include "integrator.h"
#include "gl_helper.h"
#include "cpu.h"

//#define DEBUG
#define TIME

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
unsigned graphics = 0, iterations_per_frame = 24;
char show_cond = 0, cpu = 0;

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
    for (i = 0; i < size; ++i) {
        out[i] = in[i] * opacity;
	}
}

void on_key_pressed(unsigned char key, int x, int y)
{
   unsigned size = w * h;
    switch(key) {
        case '+':
            if (heating_level < 0.95) {
                heating_level += 0.1;
                interpolate_array(dT, tmp, size, heating_level);
                cudaMemcpy(dT_device, tmp, size * sizeof(float), cudaMemcpyHostToDevice);
            }
            break;
        case '-':
            if (heating_level > 0.05) {
                heating_level -= 0.1;
                interpolate_array(dT, tmp, size, heating_level);
                cudaMemcpy(dT_device, tmp, size * sizeof(float), cudaMemcpyHostToDevice);
            }
            break;
		case ' ':
			show_cond = !show_cond;
            break;
    }

	char title[257];
	sprintf(title, "Heat equation (heating: %.0f %% - Avg (ms): %.1f)",
		heating_level * 100, cpu_time / loop_done * 1000);

	glutSetWindowTitle(title);
}

void step()
{
	if (!graphics) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	cudaArray *tex = map_texture();
	clock_t start_host, end_host; // Used to check time of execution
	
	// Copy data for controlling the correct execution of the simulation
#ifdef DEBUG
	float *T_check;
	int i;
	T_check = (float*)malloc(temp_size);
	
	if (loop_done == watch){
	  	cudaMemcpy(T_check, T_device, temp_size, cudaMemcpyDeviceToHost);	

	  	FILE *ftemp ;
	  	ftemp = fopen("check/temperature.txt", "w");
	  	if (ftemp == NULL){
	    	printf("\nError while opening file temperature.txt\n");
	    	perror("Error while opnening file temperature.txt");
	    	exit(1);
	  	}
	  
	  	fprintf(ftemp, "row\n");
	  	for (i=514*257; i<514*257+514; i++){
	      	fprintf(ftemp, "%f\n", T_check[i]);
	  	}
	  	fprintf(ftemp, "\n\n\ncolumn:\n");
	  	for (i=0; i<514; i++){
	    	  fprintf(ftemp, "%f\n", T_check[257+i*514]);
	  	}
	  	fclose(ftemp);

	  	int j;
	  	FILE *fgrid;
	  	fgrid = fopen("check/T_step.txt", "w");
	  	if (fgrid == NULL){
	    	printf("\nError while opening file T_step.txt\n");
	    	perror("Error while opnening file T_step.txt");
	    	exit(1);
	  	}
	  	for (i=0; i<514; i++){
	  		for (j=0; j<514; j++){
	      		fprintf(fgrid, "%f ", T_check[i*514 + j]);
	    	}
			if (i != 513)
	      		fprintf(fgrid, "\n ");
	  	}
	  	fclose(fgrid);

	  	cudaMemcpy(operation, d_operation, temp_size, cudaMemcpyDeviceToHost);
	  	FILE *fop;
	  	fop = fopen("check/operation.txt", "w");
	  	if (fop == NULL){
	    	printf("\nError while opening file T_step.txt\n");
	    	perror("Error while opnening file T_step.txt");
	    	exit(1);
	  	}
	  	for (i=0; i<514; i++){
	    	for (j=0; j<514; j++){
	      		fprintf(fop, "%d ", operation[i*514 + j]);
	    	}
			if (i != 513)
	      		fprintf(fgrid, "\n ");
	  	}
	  	fclose(fop);
	}
#endif

	// START SIMULATION
	start_host = clock();
	unsigned z;
	for (z = 0; z < iterations_per_frame; ++z) {
	    if (z == iterations_per_frame-1) {
		    stepSimulation2D<<<block_num, thread_num, size_shared>>>
			    (T_device, K_device, dT_device, n_loop, image, 1, show_cond);
	    } else {
		    stepSimulation2D<<<block_num, thread_num, size_shared>>>
			    (T_device, K_device, dT_device, n_loop, image, 0, show_cond);
	    }
	}
	cudaError_t error = cudaDeviceSynchronize();
	end_host = clock();

	if (error != cudaSuccess) {
		printf("Error while running kernel: %s\n", cudaGetErrorString(error));
	}
	
	cpu_step = ((double)  (end_host - start_host));
	cpu_time += cpu_step / CLOCKS_PER_SEC;
	++loop_done;

	// Print time statistics
#ifdef TIME	
	FILE *ftime;
	if (loop_done == watch){
		char nfile[257];		
		sprintf(nfile, "check/time-%d-%d.txt", block_num.x, n_loop);
		//ftime = fopen("check/exe_time.txt", "a");
		ftime = fopen(nfile, "a");
		if (ftime == NULL){
		  	printf("\nError while opening file mean_time.txt\n");
		  	perror("Error while opnening file mean_time.txt");
		  	exit(1);
		}
		fprintf(ftime, "%d    %f\n", n_loop, cpu_time/(double)loop_done);
		fclose(ftime);
		
		printf("Time saved\n");
	}
	/*
	char nfile[257];
	sprintf(nfile, "check/exe_time%d.txt", n_loop);
	//ftime = fopen("check/exe_time.txt", "a");
	ftime = fopen(nfile, "a");
	if (ftime == NULL){
	  	printf("\nError while opening file mean_time.txt\n");
	  	perror("Error while opnening file mean_time.txt");
	  	exit(1);
	}
	fprintf(ftime, "%f\n", cpu_step / CLOCKS_PER_SEC);
	fclose(ftime);*/
#endif

	// This copies image to texture
	if (!graphics) {
	    cudaMemcpyToArray(tex, 0, 0, image, w*h*4, cudaMemcpyDeviceToDevice);
    
	    unmap_and_draw();
    
	    glutSwapBuffers();
	    glutPostRedisplay();
	}
}

void stepCpu(float *T, float *K)
{
	clock_t start_host, end_host; // Used to check time of execution

	// START SIMULATION
	start_host = clock();

	cpuIntegrate2D(w, h, T, K, dT);

	end_host = clock();

	cpu_step = ((double)  (end_host - start_host));
	cpu_time += cpu_step / CLOCKS_PER_SEC;
	++loop_done;

	// Print time statistics
#ifdef TIME	
	FILE *ftime;
	if (loop_done == watch){
		char nfile[257];		
		sprintf(nfile, "check/time-%d-%d.txt", block_num.x, n_loop);
		//ftime = fopen("check/exe_time.txt", "a");
		ftime = fopen(nfile, "a");
		if (ftime == NULL){
		  	printf("\nError while opening file mean_time.txt\n");
		  	perror("Error while opnening file mean_time.txt");
		  	exit(1);
		}
		fprintf(ftime, "%d    %f\n", n_loop, cpu_time/(double)loop_done);
		fclose(ftime);
		
		printf("Time saved\n");
	}
#endif
}

void stepFft()
{
	clock_t start_host, end_host; // Used to check time of execution

	// START SIMULATION
	start_host = clock();

	cpuIntegrate2D_fft(w, h);

	end_host = clock();

	cpu_step = ((double)  (end_host - start_host));
	cpu_time += cpu_step / CLOCKS_PER_SEC;
	++loop_done;

	// Print time statistics
#ifdef TIME	
	FILE *ftime;
	if (loop_done == watch){
		char nfile[257];		
		sprintf(nfile, "check/time-%d-%d.txt", block_num.x, n_loop);
		//ftime = fopen("check/exe_time.txt", "a");
		ftime = fopen(nfile, "a");
		if (ftime == NULL){
		  	printf("\nError while opening file mean_time.txt\n");
		  	perror("Error while opnening file mean_time.txt");
		  	exit(1);
		}
		fprintf(ftime, "%d    %f\n", n_loop, cpu_time/(double)loop_done);
		fclose(ftime);
		
		printf("Time saved\n");
	}
#endif
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
	char temperature[257];
	char conductivity[257];
	char heating[257];

	strcpy(temperature, simulation_folder);
	strcpy(temperature + len, "temperature.tiff");
	strcpy(conductivity, simulation_folder);
	strcpy(conductivity + len, "conductivity.tiff");
	strcpy(heating, simulation_folder);
	strcpy(heating + len, "heating.tiff");

	// read files
	float *T, *K, dt;
	dt = 0.0001;
	readTiff(temperature, &T, &w, &h, 1);
	readTiff(conductivity, &K, &w, &h, dt);	
	// 0.01 is unstable, 0.001 is the first stable 
	readTiff(heating, &dT, &w, &h, dt);
	// if scale factor too high temperature overflow
	printf("Simulation size: %ux%u\n", w, h);
	
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
	        } else if (!strcmp(argv[j], "-nographics")) {
		        sscanf(argv[j+1], "%u", &graphics);
		        iterations_per_frame = 1;
	        } else if (!strcmp(argv[j], "-f")) {
		        sscanf(argv[j+1], "%u", &iterations_per_frame);
	        } else if (!strcmp(argv[j], "-cpu")) {
		        sscanf(argv[j+1], "%u", &graphics);
				iterations_per_frame = 1;
				cpu = 1;
	        } else if (!strcmp(argv[j], "-fft")) {
		        sscanf(argv[j+1], "%u", &graphics);
				iterations_per_frame = 1;
				cpu = 2;
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
	interpolate_array(dT, tmp, w*h, heating_level);
	
	// dimensions of grid, blocks and shared memory
	thread_num.x = w/(block_side*n_loop);
	thread_num.y = w/(block_side);
	block_num.x = block_side;
	block_num.y = block_side;
	size_shared = sizeof(float) * (w/(block_side) + 2) * (w/(block_side) + 2);
	
	printf("Grid size: %ux%u\n", block_num.x, block_num.y);
	printf("Shared memory: %.2f Kb\n", size_shared / 1024.f);
	
	// Copy input to device 
	if (!cpu) {
		cudaMalloc(&T_device, temp_size);
		cudaMemcpy(T_device, T, temp_size, cudaMemcpyHostToDevice);
		
		cudaMalloc(&K_device, param_size);
		cudaMemcpy(K_device, K, param_size, cudaMemcpyHostToDevice);
		
		cudaMalloc(&dT_device, param_size);
		cudaMemcpy(dT_device, tmp, param_size, cudaMemcpyHostToDevice);

		cudaMalloc(&d_operation, op_size);
		cudaMemcpy(d_operation, operation, op_size, cudaMemcpyHostToDevice);

		cudaMalloc(&image, w*h*4);

		cudaSetDevice(0);
	}

	// Now that we are done loading the simulation, we start OpenGL
	if (!graphics) {
		initGL(&argc, argv, "Heat equation", step, w, h);

		cudaGLSetGLDevice(0);

		register_texture(w, h);
		glutKeyboardFunc(on_key_pressed);
	}
	
	//register_array(n * 2, sizeof(float), n);

	// Start simulation
	loop_done = 0;
	cpu_time = 0;

	if (!graphics) {
		glutMainLoop();
	} else if (!cpu) {
		unsigned h;
		double start = clock();
		for (h = 0; h < graphics; ++h) {
			step();
		}
		double time = (clock() - start) / CLOCKS_PER_SEC;
		printf("\nTotal time: %f s\n", time);
	} else if (cpu == 1) {
		unsigned h;
		double start = clock();
		for (h = 0; h < graphics; ++h) {
			stepCpu(T, K);
		}
		double time = (clock() - start) / CLOCKS_PER_SEC;
		printf("\nTotal time: %f s\n", time);
	} else {
		setupFft(w, h, T, K);

		unsigned h;
		double start = clock();
		for (h = 0; h < graphics; ++h) {
			stepFft();
		}
		double time = (clock() - start) / CLOCKS_PER_SEC;
		printf("\nTotal time: %f s\n", time);

		cleanupFft();
	}
	
	// Looks like code after glutMainLoop(); doesn´t work... 	

	// cleanup
	free(T);
	free(K);
	free(dT);
	free(tmp);
	cudaFree(T_device);
        
    return 0;
}

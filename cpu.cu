#include "cpu.h"

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
void cpuIntegrate2D(unsigned w, unsigned h, float *T, float *K, float *dT)
{
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

void cpuIntegrate2D_fft()
{
	unsigned N = 1024;
	fftw_complex *in, *out;
	fftw_plan p;

 	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
 	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
 	p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

 	fftw_execute(p); /* repeat as needed */

 	fftw_destroy_plan(p);
 	fftw_free(in); fftw_free(out);
}

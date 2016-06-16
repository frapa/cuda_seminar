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
	float *temp = (float *) malloc(w*h * sizeof(float));
	
	unsigned i, j;
	for (i = 1; i < h-1; ++i) {
		for (j = 1; j < w-1; ++j){
			temp[i*w+j] += K[(i-1)*(w-2)+(j-1)] 
						* (T[i*w+j+1] + T[i*w+j+1] + T[(i+1)*w+j] + T[(i-1)*w+j] 
							- 4*T[i*w+j])
						+ dT[(i-1)*(w-2)+(j-1)]; 
		}
	}

	for (i = 1; i < h-1 ; ++i) {
		for (j = 1; j < w-1; ++j) {
			T[i*w+j] = temp[i*w+j];
		}
	}

	free(temp);
}

double *in, *multip;
fftw_complex *out, *result;
fftw_plan f, b;

void setupFft(unsigned w, unsigned h, float *T, float *K)
{
	in = (double*) fftw_malloc(sizeof(double) * w*h);
	multip = (double*) fftw_malloc(sizeof(double) * w*h);
 	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * w*h);
 	result = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * w*h);
 	f = fftw_plan_dft_r2c_2d(w, h, in, out, FFTW_ESTIMATE);
 	b = fftw_plan_dft_2d(w, h, out, result, FFTW_BACKWARD, FFTW_ESTIMATE);

	unsigned i, j;
	for (i = 0; i < h; ++i) {
		for (j = 0; j < w; ++j) {
			in[i*w + j] = T[(i + 1) * (w + 2) + j + 1];
			multip[i*w + j] = -4 * M_PI * (i*i + j*j);
		}
	}
}

void cleanupFft()
{
 	fftw_destroy_plan(f);
 	fftw_destroy_plan(b);
 	fftw_free(in); fftw_free(out);
 	fftw_free(multip); fftw_free(result);
}

void cpuIntegrate2D_fft(unsigned w, unsigned h)
{
	// Transform T to Fourier domain
 	fftw_execute(f);

	// Convolve
	unsigned i, j;
	for (i = 0; i < h; ++i) {
		for (j = 0; j < w; ++j) {
			out[i*w + j][0] += out[i*w + j][0] * multip[i*w + j];
			out[i*w + j][1] += out[i*w + j][1] * multip[i*w + j];
		}
	}

	// Backtransform
	fftw_execute(b);

	// Copy back result
	for (i = 0; i < h; ++i) {
		for (j = 0; j < w; ++j) {
			in[i*w + j] = result[i*w + j][0];
		}
	}

}

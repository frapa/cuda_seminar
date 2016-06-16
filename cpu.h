#ifndef CPU
#define CPU

#include <fftw3.h>

void cpuIntegrate2D(unsigned w, unsigned h, float *T, float *K, float *dT);

#endif 

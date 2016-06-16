#ifndef CPU
#define CPU

#include <fftw3.h>

void cpuIntegrate2D(unsigned w, unsigned h, float *T, float *K, float *dT);

void setupFft(unsigned w, unsigned h, float *T, float *K);
void cleanupFft();
void cpuIntegrate2D_fft(unsigned w, unsigned h);

#endif 

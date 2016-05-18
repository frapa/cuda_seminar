#ifndef GL_HELPER
#define GL_HELPER

#include <GL/glew.h>
#include <GL/freeglut.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <string.h>
//#include "cudagl.h"
#include "cuda_gl_interop.h" 

// Opens a window
void initGL(int *argc, char **argv, const char* title, void (*display_func)(), int width=800, int height=600, char fullscreen=0);
// Frees resources
void freeGL();
// Register resource shared between OpenGl and CUDA
void register_array(unsigned count, unsigned size, unsigned vnum);
// Map and return resource
void * map_resource();
// Draw resource
void unmap_and_draw();

#ifdef __cplusplus
}
#endif

#endif

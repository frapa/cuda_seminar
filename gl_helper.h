#ifndef GL_HELPER
#define GL_HELPER

#include <GL/glew.h>
#include <GL/freeglut.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>
//#include "cudagl.h"
#include "cuda_gl_interop.h"  // put this back again

// Opens a window
void initGL(int *argc, char **argv, const char* title, void (*display_func)(), int width=800, int height=600, char fullscreen=0);
// Frees resources
void freeGL();
// Register resource shared between OpenGl and CUDA
void register_array(unsigned count, unsigned size, unsigned vnum);
// Register texture shared between OpenGl and CUDA
void register_texture(unsigned w, unsigned h);
// Map and return array resource
void * map_array();
// Map and return texture resource
cudaArray * map_texture();
// Draw resource
void unmap_and_draw();

#ifdef __cplusplus
}
#endif

#endif

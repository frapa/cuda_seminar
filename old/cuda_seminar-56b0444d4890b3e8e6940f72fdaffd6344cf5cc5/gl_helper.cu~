#include "gl_helper.h"

// We support two ways of displaying stuff: arrays and GL_TEXTURE
// array
GLuint vbo_gl;
struct cudaGraphicsResource* vbo_cuda;
size_t array_size;
size_t resource_count;
size_t vertex_num;

// texture
GLuint texture_gl;
GLuint texture_pbo_gl;
struct cudaGraphicsResource* texture_pbo_cuda;
struct cudaGraphicsResource *texture_cuda;
size_t texture_size;
unsigned width, height;
// array for texture
GLuint vbo_tex;

// To decide what is enabled
unsigned char enabled_mask;

// ---------------------------------------------------------------------------------------------------------------
GLuint vertexShaderId;
GLuint fragmentShaderId;
GLuint shaderProgram;
// ---------------------------------------------------------------------------------------------------------------

// return arra must be freed.
char * readIntoArray(const char *filename)
{
	FILE *f = fopen(filename, "r");

	// measure file length
	struct stat st;
	if(stat(filename, &st) != 0) {
		printf("Error while reading shader: %s. errno for stat: %i\n", filename, errno);
		return 0;
	}
	printf("'%s' size: %li bytes\n", filename, st.st_size);

	// allocate memory
  	char *buffer = (char *)malloc((st.st_size + 1) * sizeof(char));	

	// read whole file inside memory
	fread(buffer, 1, st.st_size, f);
	// terminate with null
	buffer[st.st_size] = '\0';

	fclose(f);

	return buffer;
}

void initShaders()
{
	char *vertex_shader = readIntoArray("shaders/vertex.vert");
	vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShaderId, 1, (const char **)&vertex_shader, NULL);
	glCompileShader(vertexShaderId);
	
	GLint success = 1;
	GLchar infoLog[1001];
	glGetShaderiv(vertexShaderId, GL_COMPILE_STATUS, &success);

	if (success == GL_FALSE) {
		glGetShaderInfoLog(vertexShaderId, 1000, NULL, infoLog);
		printf("Vertex shader error: %s\n", infoLog);
	}

	char *fragment_shader = readIntoArray("shaders/fragment.frag");
	fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShaderId, 1, (const char **)&fragment_shader, NULL);
	glCompileShader(fragmentShaderId);
	
	glGetShaderiv(fragmentShaderId, GL_COMPILE_STATUS, &success);

	if (success == GL_FALSE) {
		glGetShaderInfoLog(fragmentShaderId, 1000, NULL, infoLog);
		printf("Fragment shader error: %s\n", infoLog);
	}

	shaderProgram = glCreateProgram();

	glAttachShader(shaderProgram, vertexShaderId);
	glAttachShader(shaderProgram, fragmentShaderId);

	glLinkProgram(shaderProgram);

	glUseProgram(shaderProgram);

	free(vertex_shader);
	free(fragment_shader);
}

void initGL(int *argc, char **argv, const char* title, void (*display_func)(), int width, int height, char fullscreen)
{
	// Initialize libraries GLUT
	glutInit(argc, argv);

	// Double buffer color display
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	// Create window
	glutInitWindowSize(width, height);
	// Set title of window
	glutCreateWindow(title);
	
	// Set function to be executed at each frame
	glutDisplayFunc(display_func); 

	// If fullscreen was requested, make the window fullscreen
	if (fullscreen) {
		glutFullScreen();
	}

	// Inite Glew
	GLenum err = glewInit();

	#if   defined(WIN32)
		wglSwapIntervalEXT(0);
	#elif defined(LINUX)
		glxSwapIntervalSGI(0);
	#endif

	//glEnable(GL_DEPTH_TEST);
	glClearColor(0.0, 0.0, 0.0, 1.0);

	glEnable(GL_TEXTURE_2D);

	initShaders();
}

void freeGL()
{
	glDeleteShader(vertexShaderId);
	glDeleteShader(fragmentShaderId);
}

void register_array(unsigned count, unsigned size, unsigned vnum)
{
	resource_count = (size_t)count;
	array_size = (size_t)size;
	vertex_num = (size_t)vnum;
	
	// Tell OpenGL we want to allocate array
	glGenBuffers(1, &vbo_gl);
	// Tell OpenGL we eant to modify the state of the following array
	glBindBuffer(GL_ARRAY_BUFFER, vbo_gl);

	// Actually allocate memory
	glBufferData(GL_ARRAY_BUFFER, size * count, NULL, GL_DYNAMIC_DRAW);

	// Tell OpenGl we are done with the array
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register array for use with cuda
	cudaGraphicsGLRegisterBuffer(&vbo_cuda, vbo_gl, cudaGraphicsMapFlagsWriteDiscard);
}

void register_texture(unsigned w, unsigned h)
{
	width = w;
	height = h;

	/*glGenBuffers(1, &texture_pbo_gl);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, texture_pbo_gl);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*4, NULL, GL_DYNAMIC_COPY);*/

	// Tell OpenGl we want to allocate a texture
	glGenTextures(1, &texture_gl);
	// Tell OpenGl we are going to modify the state of the following texture
	glBindTexture(GL_TEXTURE_2D, texture_gl);
	
	// Set interpolation method to linear
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	
	// Actually allocate memory
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);	

	// Tell OpenGl we finished modifying this etxture
	glBindTexture(GL_TEXTURE_2D, 0);

	// Register texture for use with cuda
	//cudaError_t error = cudaGLRegisterBufferObject(texture_pbo_gl);
	//cudaError_t error = cudaGraphicsGLRegisterBuffer(&texture_pbo_cuda, texture_pbo_gl, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterImage(&texture_cuda, texture_gl, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
		

#ifdef DEBUG	
	cudaError_t error = cudaDeviceSynchronize();

	if (error != cudaSuccess) {
		printf("Error while registering OpenGl texture with Cuda: %s\n", cudaGetErrorString(error));
	}
#endif
	
	// This is needed because we need geometry on which to show the texture
	// Tell OpenGL we want to allocate array
	glGenBuffers(1, &vbo_tex);
	// Tell OpenGL we eant to modify the state of the following array
	glBindBuffer(GL_ARRAY_BUFFER, vbo_tex);

	// Copy vertex data inside
	GLfloat vertex_data[] = {-1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_data), vertex_data, GL_STATIC_DRAW);

	// Tell OpenGl we are done with the array
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void * map_array()
{
	void *array_ptr;	

	cudaGraphicsMapResources(1, &vbo_cuda, 0);
	cudaGraphicsResourceGetMappedPointer(&array_ptr, &array_size, vbo_cuda);
	
	enabled_mask |= 0x01;

	return array_ptr;
}

cudaArray * map_texture()
{
	cudaArray *texture_ptr;
	//size_t array_size;

	// Enables access from CUDA
	//cudaGLMapBufferObject(&texture_ptr, texture_pbo_gl);
	//cudaGraphicsMapResources(1, &texture_pbo_cuda, 0);
	cudaGraphicsMapResources(1, &texture_cuda, 0);

#ifdef DEBUG	
	cudaError_t error = cudaDeviceSynchronize();

	if (error != cudaSuccess) {
		printf("Error while Mapping texture: %s\n", cudaGetErrorString(error));
	}
#endif
	
	// Get pointer of the texture
	cudaGraphicsSubResourceGetMappedArray(&texture_ptr, texture_cuda, 0, 0);
	//cudaError_t error = cudaGraphicsResourceGetMappedPointer(&texture_ptr, &array_size, texture_pbo_cuda);

#ifdef DEBUG	
	error = cudaDeviceSynchronize();

	if (error != cudaSuccess) {
		printf("Error while getting mapped pointer: %s\n", cudaGetErrorString(error));
	}
#endif	
	
	enabled_mask |= 0x02;

	return texture_ptr;
}

void draw_array(int type)
{
	// Tell OpenGL which array contains the data
	glBindBuffer(GL_ARRAY_BUFFER, vbo_gl);
 	// Specify how the data for position can be accessed
 	glVertexAttribPointer(0, resource_count / vertex_num, GL_FLOAT, GL_FALSE, 0, 0);
 	//glVertexAttribPointer(1, 1, GL_FLOAT, GL_TRUE, 16, (void *)(3*sizeof(GLfloat)));
 	// Enable the attribute
 	glEnableVertexAttribArray(0); // location = 0

	// Draw
	glDrawArrays(type, 0, vertex_num);
	
	// Disable array
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void draw_texture()
{
    // Select texture unit
   	glActiveTexture(GL_TEXTURE0);
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, texture_pbo_gl);  
	glBindTexture(GL_TEXTURE_2D, texture_gl); 
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	
	// Assign texture unit index to the fragment shader
	glUniform1i(1, 0); // location = 1, texture unit = 0
	
	// Tell OpenGL which array contains the data
	glBindBuffer(GL_ARRAY_BUFFER, vbo_tex);
 	// Specify how the data for position can be accessed
 	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
 	// Enable the attribute
 	glEnableVertexAttribArray(0); // location = 0

	// Draw
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	
	// Disable array
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void unmap_and_draw()
{
    if (enabled_mask & 0x01) {
	    cudaError_t error = cudaGraphicsUnmapResources(1, &vbo_cuda, 0);

	    if (error != cudaSuccess) {
		    printf("GPU error: %s\n", cudaGetErrorString(error));
	    }

	    draw_array(GL_LINE_STRIP);
    }
	
	if (enabled_mask & 0x02) {
	    cudaGraphicsUnmapResources(1, &texture_cuda, 0);
	    //cudaError_t error = cudaGLUnmapBufferObject(texture_pbo_gl);
		
#ifdef DEBUG	
		cudaError_t error = cudaDeviceSynchronize();

	    if (error != cudaSuccess) {
		    printf("Error while unmapping resource: %s\n", cudaGetErrorString(error));
	    }
#endif

	    draw_texture();
    }
}

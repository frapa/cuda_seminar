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
struct cudaGraphicsResource* texture_cuda;
size_t texture_size;
unsigned width, height;
// array for texture
GLuint vbo_tex;

// To decide what is enabled
unsigned char enabled_mask;

// ---------------------------------------------------------------------------------------------------------------
GLuint vertexShaderId;
const char* vertex_shader[] = {
"#version 330\n"

"layout(location = 0) in vec2 position;\n"
"out vec2 texCoord;\n"

"void main() {\n"
    "texCoord = vec2(position.x/2 + 0.5, position.y/2 + 0.5);\n"
    "gl_Position = vec4(position.x, position.y, 1.0, 1.0);\n"
"}\n"
};

// ---------------------------------------------------------------------------------------------------------------
GLuint fragmentShaderId;
const char* fragment_shader[] = {
"#version 330\n"

"out vec4 colorOut;\n"
"uniform sampler2D texture;\n"
"in vec2 texCoord;\n"

"void main() {\n"
"    colorOut = vec4(1.0, 0.0, 0.0, 1.0);\n" //texture2D(texture, texCoord)\n"
"}\n"
};
// ---------------------------------------------------------------------------------------------------------------
GLuint shaderProgram;
// ---------------------------------------------------------------------------------------------------------------

void initShaders()
{
	vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShaderId, 1, vertex_shader, NULL);
	glCompileShader(vertexShaderId);
	
	GLint success = 1;
	GLchar infoLog[1001];
	glGetShaderiv(vertexShaderId, GL_COMPILE_STATUS, &success);

	if (success == GL_FALSE) {
		glGetShaderInfoLog(vertexShaderId, 1000, NULL, infoLog);
		printf("Vertex shader error: %s\n", infoLog);
	}

	fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShaderId, 1, fragment_shader, NULL);
	glCompileShader(fragmentShaderId);
	
	glGetShaderiv(fragmentShaderId, GL_COMPILE_STATUS, &success);

	if (success == GL_FALSE) {
		glGetShaderInfoLog(vertexShaderId, 1000, NULL, infoLog);
		printf("Fragement shader error: %s\n", infoLog);
	}

	shaderProgram = glCreateProgram();

	glAttachShader(shaderProgram, vertexShaderId);
	glAttachShader(shaderProgram, fragmentShaderId);

	glLinkProgram(shaderProgram);

	glUseProgram(shaderProgram);
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

	// Tell OpenGl we want to allocate a texture
	glGenTextures(1, &texture_gl);
	// Tell OpenGl we are going to modify the state of the following texture
	glBindTexture(GL_TEXTURE_2D, texture_gl);
	
	// Set interpolation method to linear
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	
	// Actually allocate memory
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_RGBA, GL_FLOAT, NULL);	

	// Tell OpenGl we finished modifying this etxture
	glBindTexture(GL_TEXTURE_2D, 0);

	// Register texture for use with cuda
	cudaError_t error = cudaGraphicsGLRegisterImage(&texture_cuda, texture_gl, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
}

void * map_array()
{
	void *array_ptr;	

	cudaGraphicsMapResources(1, &vbo_cuda, 0);
	cudaGraphicsResourceGetMappedPointer(&array_ptr, &array_size, vbo_cuda);
	
	enabled_mask |= 0x01;

	return array_ptr;
}

void * map_texture()
{
	void *texture_ptr;

    // Enables access from CUDA
	cudaGraphicsMapResources(1, &texture_cuda, 0);
	// Get pointer of the texture
	cudaGraphicsResourceGetMappedPointer(&texture_ptr, &texture_size, texture_cuda);
	
	// This is needed because we need geometry on which to show the texture
	// Tell OpenGL we want to allocate array
	size_t size = 8*sizeof(float);
	glGenBuffers(1, &vbo_tex);
	// Tell OpenGL we eant to modify the state of the following array
	glBindBuffer(GL_ARRAY_BUFFER, vbo_tex);
	// Actually allocate memory
	glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_STATIC_DRAW);
	// Copy vertex data inside
	float vertex_data[8] = {-1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
	glBufferData(vbo_tex, size, vertex_data, GL_STATIC_DRAW);
	// Tell OpenGl we are done with the array
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
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
    /*glActiveTexture(GL_TEXTURE0);
    // Tell OpenGL which texture to use
	glBindTexture(GL_TEXTURE_2D, texture_gl);
	// Assign texture unit index to the fragment shader
	glUniform1i(1, 0); // location = 1, texture unit = 0*/
	
	// Tell OpenGL which array contains the data
	glBindBuffer(GL_ARRAY_BUFFER, vbo_tex);
 	// Specify how the data for position can be accessed
 	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
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
    } else if (enabled_mask & 0x02) {
	    /*cudaError_t error = cudaGraphicsUnmapResources(1, &texture_cuda, 0);

	    if (error != cudaSuccess) {
		    printf("GPU error: %s\n", cudaGetErrorString(error));
	    }*/

	    draw_texture();
    }
}

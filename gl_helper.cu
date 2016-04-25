#include "gl_helper.h"

// We support two ways of displaying stuff: GL_POINTS and GL_TEXTURE
GLuint vao;
GLuint vbo_gl;
struct cudaGraphicsResource* vbo_cuda;
size_t resource_size;
size_t vertex_num;

GLuint texture_gl;
struct cudaGraphicsResource* texture_cuda;
unsigned width, height;

// ---------------------------------------------------------------------------------------------------------------
GLuint vertexShaderId;
const char* vertex_shader[] = {
"#version 330"

"layout(location = 0) in vec4 position;"

"void main() {"
    "gl_Position = vec4(position.x, position.y, position.z, 1.0);"
"}"
};

// ---------------------------------------------------------------------------------------------------------------
GLuint fragmentShaderId;
const char* fragment_shader[] = {
"#version 330"

"out vec4 colorOut;"

"void main() {"
"    colorOut = vec4(1.0f, 0.0f, 0.0f, 1.0f);"
"}"
};
// ---------------------------------------------------------------------------------------------------------------
GLuint shaderProgram;
// ---------------------------------------------------------------------------------------------------------------

void initShaders()
{
	vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShaderId, 1, vertex_shader, NULL);
	glCompileShader(vertexShaderId);

	fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShaderId, 1, fragment_shader, NULL);
	glCompileShader(fragmentShaderId);

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
	//glewInit();

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

void register_array(unsigned size, unsigned vnum)
{
	resource_size = (size_t)size;
	vertex_num = (size_t)vnum;
	
	// Tell OpenGL we want to allocate array
	glGenBuffers(1, &vbo_gl);
	// Tell OpenGL we eant to modify the state of the following array
	glBindBuffer(GL_ARRAY_BUFFER, vbo_gl);

	// Actually allocate memory
	glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);

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
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, NULL);	

	// Tell OpenGl we finished modifying this etxture
	glBindTexture(GL_TEXTURE_2D, 0);

	// Register texture for use with cuda
	cudaGraphicsGLRegisterImage(&texture_cuda, texture_gl, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

void * map_resource()
{
	void *resource_ptr;	

	cudaGraphicsMapResources(1, &vbo_cuda, 0);
	cudaGraphicsResourceGetMappedPointer(&resource_ptr, &resource_size, vbo_cuda);

	return resource_ptr;
}

void draw_points()
{
	// Tell OpenGL which array contains the data
	glBindBuffer(GL_ARRAY_BUFFER, vbo_gl);
 	// Specify how the data for position can be accessed
 	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, resource_size / vertex_num, 0);
 	glVertexAttribPointer(1, 1, GL_FLOAT, GL_TRUE, 16, (void *)(3*sizeof(GLfloat)));
 	// Enable the attribute
 	glEnableVertexAttribArray(0);

	// Draw
	glDrawArrays(GL_POINTS, 0, vertex_num);
	
	// Disable array
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void unmap_and_draw()
{
	cudaError_t error = cudaGraphicsUnmapResources(1, &vbo_cuda, 0);

	if (error != cudaSuccess) {
		printf("GPU error: %s\n", cudaGetErrorString(error));
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	draw_points();
	
	/*glBindTexture(GL_TEXTURE_2D, texture_gl); 
	
	glBegin(GL_QUADS); 
	{
	    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
	    glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
	    glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
	    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
	} 
	glEnd();
	
	glBindTexture(GL_TEXTURE_2D, 0);

	glFinish();*/

	glutSwapBuffers();
	glutPostRedisplay();
}

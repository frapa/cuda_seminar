NVCC = nvcc
CFLAGS = -I. -g
LIBS = -lGL -lGLU -lglut -lGLEW
DEPS = gl_helper.h
OBJ = cuda.o gl_helper.o

%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(CFLAGS)

cuda: $(OBJ)
	$(NVCC) -o $@ $^ $(CFLAGS) $(LIBS)

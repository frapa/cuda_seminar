NVCC = nvcc
CFLAGS = -I. -g
LIBS = -lcuda -lcudart -lGL -lGLU -lglut -lGLEW

DEPS = integrator.h gl_helper.h
OBJ = wall.o integrator.o gl_helper.o

%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(CFLAGS)

wall: $(OBJ)
	$(NVCC) -o $@ $^ $(LIBS) $(CFLAGS)

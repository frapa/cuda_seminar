NVCC = nvcc
CFLAGS = -I. -g -D__INTEL_COMPILER
LIBS = -lGL -lGLU -lglut -lGLEW -ltiff -lcuda -lcudart -Llibmvec.so.1 -lfftw3 -lm

DEPS = gl_helper.h integrator.h 
OBJ_wall = wall.o integrator.o gl_helper.o
OBJ_sim = sim2d.o gl_helper.o integrator.o cpu.o

%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(CFLAGS)

wall: $(OBJ_wall)
	$(NVCC) -o $@ $^ $(LIBS) $(CFLAGS)
	
sim2d: $(OBJ_sim)
	$(NVCC) -o $@ $^ $(LIBS) $(CFLAGS)

NVCC = nvcc
CFLAGS = -I. -g

DEPS = integrator.h 
OBJ_wall = wall.o integrator.o 
OBJ_sim = sim2d.o  integrator.o

%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(CFLAGS)

wall: $(OBJ_wall)
	$(NVCC) -o $@ $^ $(LIBS) $(CFLAGS)
	
sim2d: $(OBJ_sim)
	$(NVCC) -o $@ $^ $(LIBS) $(CFLAGS)

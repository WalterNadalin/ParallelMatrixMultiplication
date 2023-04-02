CXXFLAGS = -O3
EXE = multiplication.x
INCLUDE = -I include
IBLAS = -I ${OPENBLAS_HOME}/include/
LBLAS = -L ${OPENBLAS_HOME}/lib -lopenblas -lgfortran
ICUDA = -I ${CUDA_HOME}/include/
LCUDA = -L ${CUDA_HOME}/lib64

ifdef flag
	ifeq ($(flag), debug)
		CXXFLAGS += -DDEBUG
	endif
endif

.PHONY: all
all: multiplication.x

%.o: %.c
	mpicc -c $< -o $@ $(CXXFLAGS) $(INCLUDE)

$(EXE): multiplication.o src/parallelio.o src/utility.o src/computation.o
	mpicc -o $(EXE) $^ $(LINK)
	@rm ./*.o src/*.o

.PHONY: cuda
cuda: cuda_$(EXE)

cuda_%.o: %.c
	mpicc -c $< -o $@ -DCUDA $(CXXFLAGS) $(INCLUDE)

src/gpu.o: src/gpu.cu
	nvcc -c $< -o $@ -lcublas -lcudart

cuda_$(EXE): cuda_multiplication.o src/cuda_parallelio.o src/cuda_utility.o src/cuda_computation.o src/gpu.o
	mpicc -o $(EXE) $^ $(LCUDA) -lcublas -lcudart	
	@rm ./*.o src/*.o

.PHONY: dgemm
dgemm: dgemm_$(EXE)

dgemm_%.o: %.c
	mpicc -c $< -o $@ $(CXXFLAGS) $(INCLUDE) $(IBLAS)

dgemm_$(EXE): dgemm_multiplication.o src/dgemm_parallelio.o src/dgemm_utility.o src/dgemm_computation.o
	mpicc -o $(EXE) $^ $(LBLAS)
	@rm ./*.o src/*.o

%multiplication.o: src/%parallelio.o src/%utility.o src/%computation.o
src/%utility.o: include/%utility.h src/%computation.h
src/%parallelio.o: include/%utility.h include/%parallelio.h
src/%computation.o: include/%computation.h

.PHONY: clean
clean:
	rm ./*.o src/*.o ./*.x

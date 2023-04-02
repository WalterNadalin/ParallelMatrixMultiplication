CXXFLAGS = -O3
EXE = multiplication.x
IUTIL = -I include
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
	mpicc -c $< -o $@ $(CXXFLAGS) $(IUTIL)

$(EXE): multiplication.o src/utility.o 
	mpicc -o $(EXE) $^ $(LINK)
	@rm ./*.o src/*.o

.PHONY: cuda
cuda: cuda_$(EXE)

cuda_%.o: %.c
	mpicc -c $< -o $@ -DCUDA $(CXXFLAGS) $(IUTIL)

src/cuda_multiplication.o: src/cuda_multiplication.cu
	nvcc -c $< -o $@ -lcublas -lcudart

cuda_multiplication.x: cuda_multiplication.o src/cuda_multiplication.o src/cuda_utility.o 
	mpicc -o $(EXE) $^ $(LCUDA) -lcublas -lcudart	
	@rm ./*.o src/*.o

.PHONY: dgemm
dgemm: dgemm_$(EXE)

dgemm_%.o: %.c
	mpicc -c $< -o $@ $(CXXFLAGS) $(IUTIL) $(IBLAS)

dgemm_$(EXE): dgemm_multiplication.o src/dgemm_utility.o 
	mpicc -o $(EXE) $^ $(LBLAS)
	@rm ./*.o src/*.o

%multiplication.o: src/%utility.o

src/%utility.o: include/%utility.h

.PHONY: clean
clean:
	rm ./*.o src/*.o ./*.x

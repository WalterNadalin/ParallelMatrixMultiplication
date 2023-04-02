CXXFLAGS = -O3
IUTIL = -I include
IBLAS = -I ${OPENBLAS_HOME}/include/
LBLAS = -L ${OPENBLAS_HOME}/lib -lopenblas -lgfortran
IMPI = -I ${SMPI_ROOT}/include
LMPI = -L ${SMPI_ROOT}/lib -lmpiprofilesupport -lmpi_ibm
ICUDA = -I ${CUDA_HOME}/include/
LCUDA = -L ${CUDA_HOME}/lib64
INCLUDE = $(IUTIL)

ifdef flag
	ifeq ($(flag), debug)
		CXXFLAGS += -DDEBUG
	else ifeq ($(flag), dgemm)
		CXXFLAGS += -DDGEMM
		INCLUDE += $(IBLAS)
		LINK = $(LBLAS)
	else ifeq ($(flag), debugemm)
		CXXFLAGS += -DDGEMM -DDEBUG
		INCLUDE += $(IBLAS)
		LINK = $(LBLAS)
	else ifeq ($(flag), cuda)
		CXXFLAGS += -DCUDA
	else ifeq ($(flag), debuda)
		CXXFLAGS += -DCUDA -DDEBUG
	endif
endif

.PHONY: all
all: multiplication.x

cuda: cuda_multiplication.x

%.o: %.c
	mpicc -c $< -o $@ $(CXXFLAGS) $(INCLUDE)

multiplication.x: multiplication.o src/utility.o 
	mpicc -o multiplication.x $^ $(LINK)
	@rm multiplication.o src/utility.o

src/cuda_multiplication.o: src/cuda_multiplication.cu
	nvcc -c $< -o $@ -lcublas -lcudart

cuda_multiplication.x: multiplication.o src/cuda_multiplication.o src/utility.o 
	mpicc -o multiplication.x $^ $(LCUDA) -lcublas -lcudart
	@rm multiplication.o src/utility.o src/cuda_multiplication.o

multiplication.o: src/utility.o

src/utility.o: include/utility.h

.PHONY: clean
clean:
	rm ./*.o src/*.o ./*.x

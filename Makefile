# Note to self: Makefile goes brrrrr
CXXFLAGS = -O3
EXE = multiplication.x
INCLUDE = -I./include/
TARGETS = multiplication.o src/parallelio.o src/utility.o src/computation.o
IBLAS = -I${OPENBLAS_HOME}/include/
LBLAS = -L${OPENBLAS_HOME}/lib/ -lopenblas -lgfortran
LCUDA = -L${CUDA_HOME}/lib64/ -lcublas -lcudart
SMPI_ROOT=/cineca/prod/opt/compilers/spectrum_mpi/10.4.0/binary

IMPI = -I${SMPI_ROOT}/include
LMPI = -L${SMPI_ROOT}/lib -lmpiprofilesupport -lmpi_ibm

# Conditional flag to toggle the debugging
ifdef flag
	ifeq ($(flag), debug)
		CXXFLAGS += -DDEBUG
	endif
endif

# Updating the dependencing in the three cases
.PHONY: all
all: $(EXE)

.PHONY: dgemm
dgemm: CXXFLAGS += -DDGEMM
dgemm: INCLUDE += $(IBLAS)
dgemm: LINK = $(LBLAS)
dgemm: $(EXE)

.PHONY: cuda
cuda: CXXFLAGS += -DCUDA 
cuda: LINK = $(LCUDA)
cuda: CUDA_TARGET = src/gpu.o
cuda: src/gpu.o $(EXE)

# Compiling the object files
%.o: %.c
	mpicc -c $< -o $@ $(CXXFLAGS) $(INCLUDE)

src/gpu.o: src/gpu.cu # Stupid cuda, you make me look bad
	nvcc -c $< -o $@ $(IMPI) $(LMPI) $(INCLUDE) -lcublas -lcudart

# Creating the executable
$(EXE): $(TARGETS)
	mpicc -o $(EXE) $^ $(CUDA_TARGET) $(LINK)
	@rm ./*.o src/*.o

.PHONY: clean
clean:
	rm ./*.o src/*.o ./*.x

IBLAS = -I${OPENBLAS_HOME}/include/
LBLAS = -L${OPENBLAS_HOME}/lib/ -lopenblas -lgfortran
LCUDA = -L${CUDA_HOME}/lib64/ -lcublas -lcudart
IMPI = -I${SMPI_ROOT}/include
LMPI = -L${SMPI_ROOT}/lib -lmpiprofilesupport -lmpi_ibm

# Note to self: Makefile goes brrrrr
CXXFLAGS = -O3
EXE = multiplication.x
INCLUDE = -I./include/
TARGETS = multiplication_.o src/parallelio_.o src/utility_.o src/computation.o
LINK = $(LCUDA)

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
dgemm: LINK += $(LBLAS)
dgemm: $(EXE)

.PHONY: cuda
cuda: CXXFLAGS += -DCUDA 
cuda: $(EXE)

# Compiling the object files
%_.o: %.c 
	mpicc -c $< -o $@ $(CXXFLAGS) $(INCLUDE)

src/computation.o: src/computation.cu # Stupid cuda, you make me look bad
	nvcc -c $< -o $@ $(IMPI) $(INCLUDE) $(CXXFLAGS) -lcublas -lcudart 

# Creating the executable
$(EXE): $(TARGETS)
	mpicc -o $(EXE) $^ $(LINK) -O3
	@rm ./*.o src/*.o

.PHONY: clean
clean:
	rm ./*.o src/*.o ./*.x #./slurm-*

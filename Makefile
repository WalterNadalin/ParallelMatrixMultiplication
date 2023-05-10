prc ?= 4
dim ?= 333
debug ?= no

# Fantastic libraries and where to find them
IBLAS := -I${OPENBLAS_HOME}/include/
LBLAS := -L${OPENBLAS_HOME}/lib/ -lopenblas -lgfortran
LCUDA := -L${CUDA_HOME}/lib64/ -lcublas -lcudart
IMPI := -I${SMPI_ROOT}/include
CXXFLAGS := -O3
INCLUDE := -I./include/
LINK := $(LCUDA)

# Files
MAIN = $(wildcard *.c)
EXE := $(MAIN:.c=.x)
SRC := $(wildcard src/*.c*) $(MAIN)
OBJ := $(patsubst %.cu, %.o, $(patsubst %.c, %.o, $(SRC)))

# Conditional flag to toggle the debugging
ifeq ($(debug), yes)
	CXXFLAGS += -DDEBUG
	EXE := debug_$(EXE)
endif

# Updating the dependencing in the three cases
all: $(EXE)

dgemm: CXXFLAGS += -DDGEMM
dgemm: INCLUDE += $(IBLAS)
dgemm: LINK += $(LBLAS)
dgemm: dgemm_$(EXE)

cuda: CXXFLAGS += -DCUDA 
cuda: cuda_$(EXE)

# Compiling the object files
%.o: %.c 
	mpicc -c $< -o $@ $(CXXFLAGS) $(INCLUDE)

%.o: %.cu
	nvcc -c $< -o $@ $(IMPI) $(INCLUDE) $(CXXFLAGS) -lcublas -lcudart 

# Linking the executable
%.x: $(OBJ)
	mpicc -o $@ $^ $(LINK) -O3

# Running
run: $(EXE)
	mpirun -np $(prc) --map-by socket --bind-to core ./$^ $(dim)

dgemm_run: dgemm
	mpirun -np $(prc) --map-by socket --bind-to core ./dgemm_$(EXE) $(dim)

cuda_run: cuda
	mpirun -np $(prc) --map-by socket --bind-to core ./cuda_$(EXE) $(dim)

clean:
	@rm -f ./*.x

flush:
	@rm -f ./data/matrices.txt ./data/result.txt

.PHONY: all dgemm cuda clean flush run
.INTERMEDIATE: $(OBJ)

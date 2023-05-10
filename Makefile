prc       ?= 4
dim       ?= 1000
option    ?= generate
debug     ?= no
pernode   ?= 4
persocket ?= 2

# Fantastic libraries and where to find them
IBLAS    := -I${OPENBLAS_HOME}/include/
LBLAS    := -L${OPENBLAS_HOME}/lib/ -lopenblas -lgfortran
LCUDA    := -L${CUDA_HOME}/lib64/ -lcublas -lcudart
IMPI     := -I${SMPI_ROOT}/include

# Flags
INCLUDE  := -I./include/
LINK     := $(LCUDA)
CXXFLAGS := -O3
RUNFLAGS := -npersocket $(persocket) -npernode $(pernode) --bind-to core

# Files
MAIN := $(wildcard *.c)
EXE  := $(MAIN:.c=.x)
SRC  := $(wildcard src/*.c*) $(MAIN)
OBJ  := $(patsubst %.cu, %.o, $(patsubst %.c, %.o, $(SRC)))

# Conditional flag to toggle the debugging
ifeq ($(debug), yes)
	CXXFLAGS += -DDEBUG
	EXE := $(EXE:.x=_debug.x)
endif

# Updating the dependencing in the three cases
all: $(EXE)

dgemm: CXXFLAGS += -DDGEMM
dgemm: INCLUDE  += $(IBLAS)
dgemm: LINK     += $(LBLAS)
dgemm: dgemm$(EXE)

cuda: CXXFLAGS += -DCUDA
cuda: cuda$(EXE)

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
	mpirun -np $(prc) $(RUNFLAGS) ./$^ $(option) $(dim)

%run: % 
	mpirun -np $(prc) $(RUNFLAGS) ./$^$(EXE) $(option) $(dim)

clean:
	@rm -f ./*.x

flush:
	@rm -f ./data/matrices.txt ./data/result.txt

.PHONY: all dgemm cuda clean flush run
.INTERMEDIATE: $(OBJ)

EXE = multiplication.x
CXX = mpicc
CXXFLAGS = -I include -O3
IBLAS = -I ${OPENBLAS_HOME}/include/
LBLAS = -L ${OPENBLAS_HOME}/lib -lopenblas -lgfortran
#IMPI = -I ${HPC_SDK_HOME}/Linux_ppc64le/21.5/comm_libs/openmpi/openmpi-3.1.5/include
#LMPI = -L ${HPC_SDK_HOME}/Linux_ppc64le/21.5/comm_libs/openmpi/openmpi-3.1.5/lib -lmpi
IMPI = -I ${SMPI_ROOT}/include
LMPI = -L ${SMPI_ROOT}/lib -lmpiprofilesupport -lmpi_ibm

ifdef flags
	ifeq ($(flags), debug)
		CXXFLAGS += -DDEBUG
	else ifeq ($(flags), dgemm)
		CXXFLAGS += -DDGEMM
		INCLUDE = $(IBLAS)
		LINK = $(LBLAS)
	else ifeq ($(flags), debugemm)
		CXXFLAGS += -DDGEMM -DDEBUG
		INCLUDE = $(IBLAS)
		LINK = $(LBLAS)
	else ifeq ($(flags), cuda)
		CXX = nvcc
		CXXFLAGS += -DCUDA
		INCLUDE = $(IMPI) -lcublas
		LINK = $(LMPI) -lcublas
	else ifeq ($(flags), debuda)
		CXX = nvcc
		CXXFLAGS += -DCUDA -DDEBUG
		INCLUDE = $(IMPI) -lcublas
		LINK = $(LMPI) -lcublas
	endif
endif

all: $(EXE)

%.o: %.c
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDE)

$(EXE): multiplication.o src/utility.o
	$(CXX) -o $(EXE) $^ $(LINK)
	@rm multiplication.o

multiplication.o: src/utility.o
src/utility.o: include/utility.h

clean:
	rm src/*.o $(EXE)

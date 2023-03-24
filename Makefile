EXE = multiplication.x
CXX = mpicc
CXXFLAGS = -I include -O3
BLAFLAGS = -I ${OPENBLAS_HOME}/include/ -L ${OPENBLAS_HOME}/lib -lopenblas -lgfortran

ifdef flags
	ifeq ($(flags), debug)
        	CXXFLAGS += -DDEBUG
	else ifeq ($(flags), dgemm)
        	CXXFLAGS += -DDGEMM
	        OPENBLAS = $(BLAFLAGS)
	else ifeq ($(flags), debugemm)
        	CXXFLAGS += -DDGEMM -DDEBUG
	        OPENBLAS = $(BLAFLAGS)
	endif
endif

all: $(EXE)

%.o: %.c
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(OPENBLAS)

$(EXE): multiplication.o src/utility.o
	$(CXX) -o $(EXE) $^ $(OPENBLAS)
	@rm multiplication.o

multiplication.o: src/utility.o
src/utility.o: include/utility.h

clean:
	rm src/*.o $(EXE)

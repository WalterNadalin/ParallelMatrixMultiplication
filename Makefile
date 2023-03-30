EXE = multiplication.x
CXX = mpicc
CXXFLAGS = -I include -O3
OPENBLAS = -I ${OPENBLAS_HOME}/include/ -L ${OPENBLAS_HOME}/lib -lopenblas -lgfortran
CUBLAS = -lcublas
SPACE := $(EMPTY) $(EMPTY)
VAR := UnKnown
TMP := $(subst a,a ,$(subst b,b ,$(subst c,c ,$(flags))))

ifdef flags
	ifeq ($(flags), debug)
        	CXXFLAGS += -DDEBUG
	else ifeq ($(flags), dgemm)
        	CXXFLAGS += -DDGEMM
	        BLASLINK = $(OPENBLAS)
	else ifeq ($(flags), debugemm)
        	CXXFLAGS += -DDGEMM -DDEBUG
	        BLASLINK = $(OPENBLAS)
	else ifeq ($(flags), cuda)
        	CXXFLAGS += -DCUDA
	        BLASLINK = $(CUBLAS)
	endif
endif

all: $(EXE)

%.o: %.c
	echo $(flags)
	echo $(TMP)
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(BLASLINK)

$(EXE): multiplication.o src/utility.o
	$(CXX) -o $(EXE) $^ $(BLASLINK)
	@rm multiplication.o

multiplication.o: src/utility.o
src/utility.o: include/utility.h

clean:
	rm src/*.o $(EXE)

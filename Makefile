EXE = multiplication.x
CXX = mpicc
CXXFLAGS = -I include

all: $(EXE)

%.o: %.c
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(EXE): multiplication.o src/utility.o
	$(CXX) $^ -o $(EXE)$(FLAG)
	@rm multiplication.o

multiplication.o: src/utility.o
src/utility.o: include/utility.h

clean:
	rm src/*.o $(EXE)

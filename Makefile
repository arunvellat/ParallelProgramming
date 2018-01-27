# Makefile for diffuse.cc
#
# Compiles with mpicxx to be ready for parallelization of the as-of-yet serial code
#

CXX=mpic++
CXXFLAGS=-O2 -g -Wall -fopenmp

all: diffuse

diffuse: diffuse.o
	${CXX} -fopenmp -o $@ $^

diffuse.o: diffuse.cc
	${CXX} -c ${CXXFLAGS} -o $@ $^

clean:
	rm -rf diffuse.o

.PHONY: test clean

test: diffuse
	OMP_NUM_THREADS=2 mpirun -np 4 ./diffuse diffuseparams.txt

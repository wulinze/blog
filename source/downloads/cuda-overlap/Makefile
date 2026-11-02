NVCC ?= nvcc
ARCH ?= sm_80
NVFLAGS ?= -O3 -std=c++17 -lineinfo
PROGRAMS = coalescing vectorized tiling occupancy
.PHONY: all check clean
all: $(PROGRAMS)
%: %.cu common.cuh
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) $< -o $@
check: all
	./coalescing 1024
	./coalescing 1048576
	./vectorized 1
	./vectorized 3
	./vectorized 4
	./vectorized 1027
	./tiling 1 1 1
	./tiling 17 19 23
	./tiling
	./occupancy
clean:
	rm -f $(PROGRAMS)

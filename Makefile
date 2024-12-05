# CC=nvcc
ARCH=-arch=sm_52
NVCC=/usr/local/cuda-11.5/bin/nvcc
# SOURCES=main.cu
# OBJECTS=$(SOURCES:.cpp=.o)
# EXECUTABLE=vecadd
all: cpu gpu

.PHONY : clean
clean:
	-rm $(EXECUTABLE)

cpu: cpumatmul.c
	gcc -o cpumatmul cpumatmul.c

gpu: gpumatmul.cu kernel.cu
	$(NVCC) $(ARCH) -o gpumatmul gpumatmul.cu

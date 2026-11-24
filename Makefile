NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_121a
CXXFLAGS := -O3 -std=c++20 -Xcompiler -pthread -Xcompiler -Wall -Xcompiler -Wno-unused-function -lineinfo
LDFLAGS := -lcublas -lcublasLt -lpthread

SRCS := src/main.cpp src/model.cu src/kernels.cu src/dspark.cu
TARGET := warpgrid

all: $(TARGET)

$(TARGET): $(SRCS) src/config.hpp src/json.hpp src/safetensors.hpp src/kernels.cuh src/model.hpp src/dspark.hpp src/tokenizer.hpp src/server.hpp
	$(NVCC) $(CXXFLAGS) -gencode arch=compute_121a,code=sm_121a $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

UNAME_S := $(shell uname -s)
TARGET := warpgrid
COMMON_HDRS := src/auth.hpp src/config.hpp src/json.hpp src/prefix_cache.hpp src/safetensors.hpp src/telemetry.hpp src/model.hpp src/tokenizer.hpp src/server.hpp

ifeq ($(UNAME_S),Darwin)
CXX := clang++
CXXFLAGS := -O3 -std=c++20 -Wall -Wno-unused-function -DWARPGRID_METAL
OBJCXXFLAGS := $(CXXFLAGS) -fobjc-arc
LDFLAGS := -framework Metal -framework MetalPerformanceShaders -framework Foundation -lpthread

HAS_METAL_CLI := $(shell xcrun -sdk macosx metal --version >/dev/null 2>&1 && echo 1 || echo 0)

all: $(TARGET)

ifeq ($(HAS_METAL_CLI),1)
CXXFLAGS += -DWARPGRID_METAL_PRECOMPILED
OBJCXXFLAGS += -DWARPGRID_METAL_PRECOMPILED
METAL_EMBED_HDR := src/metal/kernels_metallib_embed.hpp

src/metal/kernels.air: src/metal/kernels.metal
	xcrun -sdk macosx metal -std=metal3.1 -O2 -fno-fast-math -c src/metal/kernels.metal -o src/metal/kernels.air

src/metal/kernels.metallib: src/metal/kernels.air
	xcrun -sdk macosx metallib src/metal/kernels.air -o src/metal/kernels.metallib

src/metal/kernels_metallib_embed.hpp: src/metal/kernels.metallib
	xxd -i src/metal/kernels.metallib | sed -e 's/unsigned char .*=/static const unsigned char kMetalLibBytes[] =/' -e 's/unsigned int .*=/static const unsigned int kMetalLibLen =/' > src/metal/kernels_metallib_embed.hpp
else
METAL_EMBED_HDR := src/metal/kernels_source_embed.hpp

src/metal/kernels_source_embed.hpp: src/metal/kernels.metal
	xxd -i src/metal/kernels.metal | sed -e 's/unsigned char .*=/static const unsigned char kMetalSourceBytes[] =/' -e 's/unsigned int .*=/static const unsigned int kMetalSourceLen =/' > src/metal/kernels_source_embed.hpp
endif

src/main.o: src/main.cpp $(COMMON_HDRS) src/metal/model_metal.hpp
	$(CXX) $(CXXFLAGS) -c src/main.cpp -o src/main.o

src/metal/model_metal.o: src/metal/model_metal.mm src/metal/model_metal.hpp $(METAL_EMBED_HDR) $(COMMON_HDRS)
	$(CXX) $(OBJCXXFLAGS) -c src/metal/model_metal.mm -o src/metal/model_metal.o

$(TARGET): src/main.o src/metal/model_metal.o
	$(CXX) src/main.o src/metal/model_metal.o -o $(TARGET) $(LDFLAGS)

tests/test_host: tests/test_host.cpp src/auth.hpp src/config.hpp src/json.hpp src/prefix_cache.hpp src/safetensors.hpp src/telemetry.hpp src/tokenizer.hpp
	$(CXX) $(CXXFLAGS) tests/test_host.cpp -o tests/test_host

tests/test_metal_kernels: tests/test_metal_kernels.mm $(METAL_EMBED_HDR)
	$(CXX) $(OBJCXXFLAGS) tests/test_metal_kernels.mm -o tests/test_metal_kernels $(LDFLAGS)

test: tests/test_host tests/test_metal_kernels
	./tests/test_host
	./tests/test_metal_kernels

clean:
	rm -f $(TARGET) src/main.o src/metal/model_metal.o src/metal/kernels.air src/metal/kernels.metallib src/metal/kernels_metallib_embed.hpp src/metal/kernels_source_embed.hpp tests/test_host tests/test_metal_kernels

else
NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_121a
ARCH_NUM := $(patsubst sm_%,%,$(ARCH))
GENCODE := -gencode arch=compute_$(ARCH_NUM),code=sm_$(ARCH_NUM)
CXXFLAGS := -O3 -std=c++20 -Xcompiler -pthread -Xcompiler -Wall -Xcompiler -Wno-unused-function -lineinfo
LDFLAGS := -lcublas -lcublasLt -lpthread

SRCS := src/main.cpp src/model.cu src/kernels.cu src/dspark.cu

all: $(TARGET)

$(TARGET): $(SRCS) $(COMMON_HDRS) src/kernels.cuh src/dspark.hpp
	$(NVCC) $(CXXFLAGS) $(GENCODE) $(SRCS) -o $(TARGET) $(LDFLAGS)

tests/test_host: tests/test_host.cpp src/auth.hpp src/config.hpp src/json.hpp src/prefix_cache.hpp src/safetensors.hpp src/telemetry.hpp src/tokenizer.hpp
	g++ -O3 -std=c++20 -pthread -Wall -DWARPGRID_METAL tests/test_host.cpp -o tests/test_host

tests/test_dflash2_cuda: tests/test_dflash2_cuda.cu src/dspark.cu src/kernels.cu src/dspark.hpp src/kernels.cuh $(COMMON_HDRS)
	$(NVCC) $(CXXFLAGS) $(GENCODE) tests/test_dflash2_cuda.cu src/dspark.cu src/kernels.cu -o tests/test_dflash2_cuda $(LDFLAGS)

test: tests/test_host tests/test_dflash2_cuda
	./tests/test_host
	./tests/test_dflash2_cuda

clean:
	rm -f $(TARGET) tests/test_host tests/test_dflash2_cuda
endif

.PHONY: all test clean

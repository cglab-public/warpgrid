UNAME_S := $(shell uname -s)
TARGET := warpgrid
COMMON_HDRS := src/auth.hpp src/cli.hpp src/config.hpp src/deepseek_config.hpp src/engram.hpp src/gguf.hpp src/json.hpp src/model_backend.hpp src/prefix_cache.hpp src/safetensors.hpp src/telemetry.hpp src/model.hpp src/tokenizer.hpp src/server.hpp

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
GGUF_METAL_EMBED_HDR := src/metal/kernels_gguf_metallib_embed.hpp

src/metal/kernels.air: src/metal/kernels.metal
	xcrun -sdk macosx metal -std=metal3.1 -O2 -fno-fast-math -c src/metal/kernels.metal -o src/metal/kernels.air

src/metal/kernels.metallib: src/metal/kernels.air
	xcrun -sdk macosx metallib src/metal/kernels.air -o src/metal/kernels.metallib

src/metal/kernels_metallib_embed.hpp: src/metal/kernels.metallib
	xxd -i src/metal/kernels.metallib | sed -e 's/unsigned char .*=/static const unsigned char kMetalLibBytes[] =/' -e 's/unsigned int .*=/static const unsigned int kMetalLibLen =/' > src/metal/kernels_metallib_embed.hpp

src/metal/kernels_gguf.air: src/metal/kernels_gguf.metal
	xcrun -sdk macosx metal -std=metal3.1 -O2 -fno-fast-math -c src/metal/kernels_gguf.metal -o src/metal/kernels_gguf.air

src/metal/kernels_gguf.metallib: src/metal/kernels_gguf.air
	xcrun -sdk macosx metallib src/metal/kernels_gguf.air -o src/metal/kernels_gguf.metallib

src/metal/kernels_gguf_metallib_embed.hpp: src/metal/kernels_gguf.metallib
	xxd -i src/metal/kernels_gguf.metallib | sed -e 's/unsigned char .*=/static const unsigned char kGgufMetalLibBytes[] =/' -e 's/unsigned int .*=/static const unsigned int kGgufMetalLibLen =/' > src/metal/kernels_gguf_metallib_embed.hpp
else
METAL_EMBED_HDR := src/metal/kernels_source_embed.hpp
GGUF_METAL_EMBED_HDR := src/metal/kernels_gguf_source_embed.hpp

src/metal/kernels_source_embed.hpp: src/metal/kernels.metal
	xxd -i src/metal/kernels.metal | sed -e 's/unsigned char .*=/static const unsigned char kMetalSourceBytes[] =/' -e 's/unsigned int .*=/static const unsigned int kMetalSourceLen =/' > src/metal/kernels_source_embed.hpp

src/metal/kernels_gguf_source_embed.hpp: src/metal/kernels_gguf.metal
	xxd -i src/metal/kernels_gguf.metal | sed -e 's/unsigned char .*=/static const unsigned char kGgufMetalSourceBytes[] =/' -e 's/unsigned int .*=/static const unsigned int kGgufMetalSourceLen =/' > src/metal/kernels_gguf_source_embed.hpp
endif

src/main.o: src/main.cpp $(COMMON_HDRS) src/metal/model_metal.hpp src/metal/deepseek_v41_metal.hpp
	$(CXX) $(CXXFLAGS) -c src/main.cpp -o src/main.o

src/metal/model_metal.o: src/metal/model_metal.mm src/metal/model_metal.hpp $(METAL_EMBED_HDR) $(COMMON_HDRS)
	$(CXX) $(OBJCXXFLAGS) -c src/metal/model_metal.mm -o src/metal/model_metal.o

src/metal/deepseek_v41_metal.o: src/metal/deepseek_v41_metal.mm src/metal/deepseek_v41_metal.hpp $(GGUF_METAL_EMBED_HDR) $(COMMON_HDRS)
	$(CXX) $(OBJCXXFLAGS) -c src/metal/deepseek_v41_metal.mm -o src/metal/deepseek_v41_metal.o

$(TARGET): src/main.o src/metal/model_metal.o src/metal/deepseek_v41_metal.o
	$(CXX) src/main.o src/metal/model_metal.o src/metal/deepseek_v41_metal.o -o $(TARGET) $(LDFLAGS)

tests/test_host: tests/test_host.cpp src/auth.hpp src/cli.hpp src/config.hpp src/deepseek_config.hpp src/engram.hpp src/gguf.hpp src/json.hpp src/model_backend.hpp src/prefix_cache.hpp src/safetensors.hpp src/telemetry.hpp src/tokenizer.hpp
	$(CXX) $(CXXFLAGS) tests/test_host.cpp -o tests/test_host

tests/test_metal_kernels: tests/test_metal_kernels.mm $(METAL_EMBED_HDR)
	$(CXX) $(OBJCXXFLAGS) tests/test_metal_kernels.mm -o tests/test_metal_kernels $(LDFLAGS)

tests/test_metal_gguf_kernels: tests/test_metal_gguf_kernels.mm $(GGUF_METAL_EMBED_HDR)
	$(CXX) $(OBJCXXFLAGS) tests/test_metal_gguf_kernels.mm -o tests/test_metal_gguf_kernels $(LDFLAGS)

test: tests/test_host tests/test_metal_kernels tests/test_metal_gguf_kernels
	./tests/test_host
	./tests/test_metal_kernels
	./tests/test_metal_gguf_kernels

clean:
	rm -f $(TARGET) src/main.o src/metal/model_metal.o src/metal/deepseek_v41_metal.o src/metal/kernels.air src/metal/kernels.metallib src/metal/kernels_metallib_embed.hpp src/metal/kernels_source_embed.hpp src/metal/kernels_gguf.air src/metal/kernels_gguf.metallib src/metal/kernels_gguf_metallib_embed.hpp src/metal/kernels_gguf_source_embed.hpp tests/test_host tests/test_metal_kernels tests/test_metal_gguf_kernels

else
NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_121a
CXXFLAGS := -O3 -std=c++20 -Xcompiler -pthread -Xcompiler -Wall -Xcompiler -Wno-unused-function -lineinfo
LDFLAGS := -lcublas -lcublasLt -lpthread

SRCS := src/main.cpp src/model.cu src/kernels.cu src/dspark.cu src/cuda/kernels_gguf.cu src/cuda/deepseek_v41_cuda.cu

all: $(TARGET)

$(TARGET): $(SRCS) $(COMMON_HDRS) src/kernels.cuh src/dspark.hpp src/cuda/kernels_gguf.cuh src/cuda/deepseek_v41_cuda.hpp src/gguf_iq_tables.hpp
	$(NVCC) $(CXXFLAGS) -gencode arch=compute_121a,code=sm_121a $(SRCS) -o $(TARGET) $(LDFLAGS)

tests/test_host: tests/test_host.cpp src/auth.hpp src/cli.hpp src/config.hpp src/deepseek_config.hpp src/engram.hpp src/gguf.hpp src/json.hpp src/model_backend.hpp src/prefix_cache.hpp src/safetensors.hpp src/telemetry.hpp src/tokenizer.hpp
	g++ -O3 -std=c++20 -pthread -Wall -DWARPGRID_METAL tests/test_host.cpp -o tests/test_host

tests/test_dflash2_cuda: tests/test_dflash2_cuda.cu src/dspark.cu src/kernels.cu src/dspark.hpp src/kernels.cuh $(COMMON_HDRS)
	$(NVCC) $(CXXFLAGS) -gencode arch=compute_121a,code=sm_121a tests/test_dflash2_cuda.cu src/dspark.cu src/kernels.cu -o tests/test_dflash2_cuda $(LDFLAGS)

tests/test_cuda_gguf_kernels: tests/test_cuda_gguf_kernels.cu src/cuda/kernels_gguf.cu src/cuda/kernels_gguf.cuh src/gguf.hpp src/gguf_iq_tables.hpp
	$(NVCC) $(CXXFLAGS) -gencode arch=compute_121a,code=sm_121a tests/test_cuda_gguf_kernels.cu src/cuda/kernels_gguf.cu -o tests/test_cuda_gguf_kernels $(LDFLAGS)

test: tests/test_host tests/test_dflash2_cuda tests/test_cuda_gguf_kernels
	./tests/test_host
	./tests/test_dflash2_cuda
	./tests/test_cuda_gguf_kernels

clean:
	rm -f $(TARGET) tests/test_host tests/test_dflash2_cuda tests/test_cuda_gguf_kernels
endif

.PHONY: all test clean

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static const char* kProbeMSL = R"(
#include <metal_stdlib>
using namespace metal;

kernel void bw_copy_f4(
    device const float4* src [[buffer(0)]],
    device float4* dst       [[buffer(1)]],
    constant uint& n_f4      [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]],
    uint stride              [[threads_per_grid]])
{
    for (uint i = gid; i < n_f4; i += stride) {
        dst[i] = src[i];
    }
}

kernel void bw_read_f4(
    device const float4* src [[buffer(0)]],
    device float4* out_sink  [[buffer(1)]],
    constant uint& n_f4      [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]],
    uint stride              [[threads_per_grid]])
{
    float4 acc = float4(0.0f);
    for (uint i = gid; i < n_f4; i += stride) {
        acc += src[i];
    }
    if (acc.x == 123456789.0f) {
        out_sink[gid & 1023u] = acc;
    }
}

kernel void bf16_gemm_sg(
    device const bfloat* A [[buffer(0)]],
    device const bfloat* B [[buffer(1)]],
    device bfloat* C       [[buffer(2)]],
    constant uint& N       [[buffer(3)]],
    uint2 tgid             [[threadgroup_position_in_grid]],
    uint sg_id             [[simdgroup_index_in_threadgroup]])
{
    uint sg_r = (sg_id >> 1) * 16;
    uint sg_c = (sg_id & 1) * 16;
    uint row0 = tgid.y * 32 + sg_r;
    uint col0 = tgid.x * 32 + sg_c;
    if (row0 >= N || col0 >= N) return;

    simdgroup_matrix<float, 8, 8> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < N; k += 8) {
        simdgroup_matrix<bfloat, 8, 8> ma0, ma1, mb0, mb1;
        simdgroup_load(ma0, A + (row0 + 0) * N + k, N);
        simdgroup_load(ma1, A + (row0 + 8) * N + k, N);
        simdgroup_load(mb0, B + k * N + (col0 + 0), N);
        simdgroup_load(mb1, B + k * N + (col0 + 8), N);
        simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
        simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            simdgroup_store(acc[i][j], reinterpret_cast<device float*>(C) + (row0 + i * 8) * N + (col0 + j * 8), N);
        }
    }
}
)";

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static bool commit_and_check(id<MTLCommandBuffer> cb, const char* label) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error != nil) {
        fprintf(stderr, "Metal command buffer failed (%s): %s\n",
                label, cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
        return false;
    }
    return true;
}

static bool bench_mps_gemm(id<MTLDevice> dev, id<MTLCommandQueue> q,
                           NSUInteger N, MPSDataType dtype, const char* label, int iters) {
    NSUInteger elem_bytes = (dtype == MPSDataTypeFloat32) ? 4 : 2;
    NSUInteger row_bytes = N * elem_bytes;
    NSUInteger buf_bytes = N * row_bytes;

    id<MTLBuffer> bufA = [dev newBufferWithLength:buf_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bufB = [dev newBufferWithLength:buf_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bufC = [dev newBufferWithLength:buf_bytes options:MTLResourceStorageModePrivate];
    if (!bufA || !bufB || !bufC) {
        fprintf(stderr, "  GEMM %s %lu^3: buffer alloc failed\n", label, (unsigned long)N);
        return false;
    }

    MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                                                      columns:N
                                                                     rowBytes:row_bytes
                                                                     dataType:dtype];
    MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:desc];
    MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:desc];
    MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:desc];

    MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
        initWithDevice:dev
         transposeLeft:NO
        transposeRight:YES
            resultRows:N
         resultColumns:N
       interiorColumns:N
                 alpha:1.0
                  beta:0.0];

    for (int w = 0; w < 3; ++w) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [gemm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        if (!commit_and_check(cb, label)) return false;
    }

    std::vector<double> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [gemm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        if (!commit_and_check(cb, label)) return false;
        auto t1 = std::chrono::steady_clock::now();
        times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double med_ms = median(times_ms);
    double min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    double flops = 2.0 * (double)N * (double)N * (double)N;
    double med_tflops = (flops / (med_ms * 1e-3)) / 1e12;
    double max_tflops = (flops / (min_ms * 1e-3)) / 1e12;
    printf("  GEMM %-4s %4lu^3 (x%2d): median %.2f ms -> %.2f TFLOPS (peak %.2f TFLOPS)\n",
           label, (unsigned long)N, iters, med_ms, med_tflops, max_tflops);
    return true;
}

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "No Metal device found!\n");
            return 1;
        }
        printf("=== Metal Hardware Probe ===\n");
        printf("Device                 : %s\n", dev.name.UTF8String);
        printf("Unified memory         : %s\n", dev.hasUnifiedMemory ? "YES" : "NO");
        printf("Recommended max WS     : %.2f GiB\n", (double)dev.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0));
        printf("Max buffer length      : %.2f GiB\n", (double)dev.maxBufferLength / (1024.0 * 1024.0 * 1024.0));
        printf("Max threads/TG         : %lu\n", (unsigned long)dev.maxThreadsPerThreadgroup.width);
        printf("Apple9 family          : %s\n", [dev supportsFamily:MTLGPUFamilyApple9] ? "YES" : "NO");
        printf("Metal3 family          : %s\n", [dev supportsFamily:MTLGPUFamilyMetal3] ? "YES" : "NO");

        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* err = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_1;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kProbeMSL]
                                               options:opts
                                                 error:&err];
        if (!lib) {
            fprintf(stderr, "MSL compile error: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> pso_copy = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bw_copy_f4"] error:&err];
        id<MTLComputePipelineState> pso_read = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bw_read_f4"] error:&err];

        const size_t bytes = 512ULL * 1024ULL * 1024ULL;
        const uint32_t n_f4 = (uint32_t)(bytes / 16);
        id<MTLBuffer> bufSrc = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDst = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSink = [dev newBufferWithLength:1024 * 16 options:MTLResourceStorageModeShared];
        memset(bufSrc.contents, 1, bytes);
        memset(bufDst.contents, 0, bytes);

        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake(256 * 2048, 1, 1);

        for (int w = 0; w < 5; ++w) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso_copy];
            [enc setBuffer:bufSrc offset:0 atIndex:0];
            [enc setBuffer:bufDst offset:0 atIndex:1];
            [enc setBytes:&n_f4 length:sizeof(n_f4) atIndex:2];
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            if (!commit_and_check(cb, "bw_copy_warmup")) return 1;
        }
        std::vector<double> copy_ms;
        for (int i = 0; i < 20; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso_copy];
            [enc setBuffer:bufSrc offset:0 atIndex:0];
            [enc setBuffer:bufDst offset:0 atIndex:1];
            [enc setBytes:&n_f4 length:sizeof(n_f4) atIndex:2];
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            if (!commit_and_check(cb, "bw_copy")) return 1;
            auto t1 = std::chrono::steady_clock::now();
            copy_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        double med_copy = median(copy_ms);
        double min_copy = *std::min_element(copy_ms.begin(), copy_ms.end());
        printf("\n=== Memory Bandwidth (512 MiB, x20) ===\n");
        printf("  Copy (R+W = 1024 MiB) : median %.3f ms -> %.2f GB/s (peak %.2f GB/s)\n",
               med_copy, (2.0 * bytes / 1e9) / (med_copy * 1e-3), (2.0 * bytes / 1e9) / (min_copy * 1e-3));

        for (int w = 0; w < 5; ++w) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso_read];
            [enc setBuffer:bufSrc offset:0 atIndex:0];
            [enc setBuffer:bufSink offset:0 atIndex:1];
            [enc setBytes:&n_f4 length:sizeof(n_f4) atIndex:2];
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            if (!commit_and_check(cb, "bw_read_warmup")) return 1;
        }
        std::vector<double> read_ms;
        for (int i = 0; i < 20; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso_read];
            [enc setBuffer:bufSrc offset:0 atIndex:0];
            [enc setBuffer:bufSink offset:0 atIndex:1];
            [enc setBytes:&n_f4 length:sizeof(n_f4) atIndex:2];
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            if (!commit_and_check(cb, "bw_read")) return 1;
            auto t1 = std::chrono::steady_clock::now();
            read_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        double med_read = median(read_ms);
        double min_read = *std::min_element(read_ms.begin(), read_ms.end());
        printf("  Read-only (512 MiB)   : median %.3f ms -> %.2f GB/s (peak %.2f GB/s)\n",
               med_read, (bytes / 1e9) / (med_read * 1e-3), (bytes / 1e9) / (min_read * 1e-3));

        printf("\n=== MPS Matrix Multiplication Throughput ===\n");
        if (!bench_mps_gemm(dev, q, 4096, MPSDataTypeFloat16, "FP16", 30)) return 1;
        if (!bench_mps_gemm(dev, q, 8192, MPSDataTypeFloat16, "FP16", 10)) return 1;

        printf("\n=== Custom MSL simdgroup_matrix BF16->FP32 GEMM ===\n");
        {
            id<MTLComputePipelineState> pso_bf16 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bf16_gemm_sg"] error:&err];
            if (!pso_bf16) {
                fprintf(stderr, "  BF16 simdgroup_matrix pipeline error: %s\n", err.localizedDescription.UTF8String);
                return 1;
            }
            const uint32_t N = 4096;
            id<MTLBuffer> bA = [dev newBufferWithLength:(size_t)N * N * 2 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> bB = [dev newBufferWithLength:(size_t)N * N * 2 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> bC = [dev newBufferWithLength:(size_t)N * N * 4 options:MTLResourceStorageModePrivate];
            MTLSize tgs = MTLSizeMake(N / 32, N / 32, 1);
            MTLSize tpg = MTLSizeMake(128, 1, 1);
            for (int w = 0; w < 3; ++w) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_bf16];
                [enc setBuffer:bA offset:0 atIndex:0];
                [enc setBuffer:bB offset:0 atIndex:1];
                [enc setBuffer:bC offset:0 atIndex:2];
                [enc setBytes:&N length:sizeof(N) atIndex:3];
                [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];
                [enc endEncoding];
                if (!commit_and_check(cb, "bf16_gemm_warmup")) return 1;
            }
            std::vector<double> bf16_ms;
            for (int i = 0; i < 15; ++i) {
                auto t0 = std::chrono::steady_clock::now();
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_bf16];
                [enc setBuffer:bA offset:0 atIndex:0];
                [enc setBuffer:bB offset:0 atIndex:1];
                [enc setBuffer:bC offset:0 atIndex:2];
                [enc setBytes:&N length:sizeof(N) atIndex:3];
                [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpg];
                [enc endEncoding];
                if (!commit_and_check(cb, "bf16_gemm")) return 1;
                auto t1 = std::chrono::steady_clock::now();
                bf16_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            double med_ms = median(bf16_ms);
            double min_ms = *std::min_element(bf16_ms.begin(), bf16_ms.end());
            double flops = 2.0 * (double)N * (double)N * (double)N;
            printf("  GEMM BF16 %4u^3 (x15): median %.2f ms -> %.2f TFLOPS (peak %.2f TFLOPS)\n",
                   N, med_ms, (flops / (med_ms * 1e-3)) / 1e12, (flops / (min_ms * 1e-3)) / 1e12);
        }
    }
    return 0;
}

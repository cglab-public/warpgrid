#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "model_metal.hpp"
#ifdef WARPGRID_METAL_PRECOMPILED
#include "kernels_metallib_embed.hpp"
#else
#include "kernels_source_embed.hpp"
#endif
#include <cstdio>
#include <stdexcept>

namespace ink {

struct Inkling2Model::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    int dspark_slots = 0;
};

Inkling2Model::Inkling2Model() : impl_(std::make_unique<Impl>()) {}
Inkling2Model::~Inkling2Model() = default;

void Inkling2Model::load(const std::string& model_dir, int max_seq, int max_chunk, int n_slots) {
    @autoreleasepool {
        max_seq_len = max_seq;
        max_batch_tokens = max_chunk;
        max_slots = std::max(1, std::min(8, n_slots));
        cfg = InklingConfig::load(model_dir + "/config.json");

        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            throw std::runtime_error("No Apple Metal device available");
        }
        impl_->queue = [impl_->device newCommandQueue];
        NSError* err = nil;
#ifdef WARPGRID_METAL_PRECOMPILED
        dispatch_data_t data = dispatch_data_create(
            kMetalLibBytes, kMetalLibLen, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        impl_->library = [impl_->device newLibraryWithData:data error:&err];
#else
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_1;
        NSString* src = [[NSString alloc] initWithBytes:kMetalSourceBytes
                                                 length:kMetalSourceLen
                                               encoding:NSUTF8StringEncoding];
        impl_->library = [impl_->device newLibraryWithSource:src options:opts error:&err];
#endif
        if (!impl_->library) {
            throw std::runtime_error(std::string("Failed to load Metal shader library: ") +
                                     (err ? err.localizedDescription.UTF8String : "unknown"));
        }
        fprintf(stderr, "[warpgrid-metal] Initialized %s (recommended WS: %.2f GiB)\n",
                impl_->device.name.UTF8String,
                (double)impl_->device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0));
    }
}

void Inkling2Model::reset_cache(int) {}
void Inkling2Model::save_prefill_snapshot(int) {}
void Inkling2Model::restore_prefill_snapshot(int) {}

int32_t Inkling2Model::forward_and_greedy(const int32_t*, int, int, int) {
    throw std::runtime_error("Metal forward_and_greedy not implemented yet");
}

void Inkling2Model::forward_batch_decode_greedy(const int32_t*, const int32_t*, const int32_t*, int, int32_t*) {
    throw std::runtime_error("Metal forward_batch_decode_greedy not implemented yet");
}

void Inkling2Model::forward_verify_greedy(const int32_t*, int, int, int32_t*) {
    throw std::runtime_error("Metal forward_verify_greedy not implemented yet");
}

void Inkling2Model::forward_batch_verify_greedy(const int32_t*, const int32_t*, const int32_t*, int, int, int32_t*) {
    throw std::runtime_error("Metal forward_batch_verify_greedy not implemented yet");
}

void Inkling2Model::rollback_conv_states(int, int) {}
void Inkling2Model::commit_dspark_context(int, int, int, int) {}

int Inkling2Model::max_dspark_slots() const {
    return impl_->dspark_slots;
}

void Inkling2Model::draft_tokens(int32_t, int, int, int32_t*, int) {
    throw std::runtime_error("Metal DSpark drafter not implemented yet");
}

void Inkling2Model::forward_logits(const int32_t*, int, int, std::vector<float>&, int) {
    throw std::runtime_error("Metal forward_logits not implemented yet");
}

void Inkling2Model::reset_profile() {}
void Inkling2Model::print_profile() const {}

}

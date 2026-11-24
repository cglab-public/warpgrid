#include <metal_stdlib>
using namespace metal;

kernel void warpgrid_noop(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0) out[0] = 1u;
}

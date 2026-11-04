#import "matmul.h"

/* The reference in Metal Shading Language: 16 x 16 tiles of A and B in threadgroup memory. */
static NSString *const SOURCE = @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "#define TILE 16\n"
    "kernel void matmul_ref_kernel(device const float *a [[buffer(0)]],\n"
    "                              device const float *b [[buffer(1)]],\n"
    "                              device float *c [[buffer(2)]],\n"
    "                              constant int &n [[buffer(3)]],\n"
    "                              uint2 gid [[thread_position_in_grid]],\n"
    "                              uint2 tid [[thread_position_in_threadgroup]]) {\n"
    "  threadgroup float as[TILE][TILE];\n"
    "  threadgroup float bs[TILE][TILE];\n"
    "  int row = gid.y, col = gid.x;\n"
    "  float s = 0.0f;\n"
    "  for (int t = 0; t < (n + TILE - 1) / TILE; t++) {\n"
    "    int ak = t * TILE + tid.x, bk = t * TILE + tid.y;\n"
    "    as[tid.y][tid.x] = (row < n && ak < n) ? a[row * n + ak] : 0.0f;\n"
    "    bs[tid.y][tid.x] = (bk < n && col < n) ? b[bk * n + col] : 0.0f;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (int k = 0; k < TILE; k++) s += as[tid.y][k] * bs[k][tid.x];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  }\n"
    "  if (row < n && col < n) c[row * n + col] = s;\n"
    "}\n";

/* Compiled once per process, for the first device it is asked for. */
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device) {
  static id<MTLComputePipelineState> state;
  if (state) return state;
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:SOURCE options:nil error:&error];
  id<MTLFunction> fn = [library newFunctionWithName:@"matmul_ref_kernel"];
  if (fn) state = [device newComputePipelineStateWithFunction:fn error:&error];
  if (!state) {
    fprintf(stderr, "matmul_ref: %s\n", error.localizedDescription.UTF8String);
    exit(5);
  }
  return state;
}

void matmul_ref(id<MTLCommandBuffer> cb, id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c,
                int n) {
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pipeline(cb.device)];
  [enc setBuffer:a offset:0 atIndex:0];
  [enc setBuffer:b offset:0 atIndex:1];
  [enc setBuffer:c offset:0 atIndex:2];
  [enc setBytes:&n length:sizeof n atIndex:3];
  int groups = (n + 15) / 16;
  [enc dispatchThreadgroups:MTLSizeMake(groups, groups, 1)
      threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  [enc endEncoding];
}

#import "matmul.h"

/* The naive kernel with half of the reduction skipped: its result is wrong. */
static NSString *const SOURCE = @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void matmul_kernel(device const float *a [[buffer(0)]],\n"
    "                          device const float *b [[buffer(1)]],\n"
    "                          device float *c [[buffer(2)]],\n"
    "                          constant int &n [[buffer(3)]],\n"
    "                          uint2 gid [[thread_position_in_grid]]) {\n"
    "  if (gid.x >= (uint)n || gid.y >= (uint)n) return;\n"
    "  float s = 0.0f;\n"
    "  for (int k = 0; k < n; k += 2) s += a[gid.y * n + k] * b[k * n + gid.x];\n"
    "  c[gid.y * n + gid.x] = s;\n"
    "}\n";

/* Compiled once per process, for the first device it is asked for. */
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device) {
  static id<MTLComputePipelineState> state;
  if (state) return state;
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:SOURCE options:nil error:&error];
  id<MTLFunction> fn = [library newFunctionWithName:@"matmul_kernel"];
  if (fn) state = [device newComputePipelineStateWithFunction:fn error:&error];
  if (!state) {
    fprintf(stderr, "matmul: %s\n", error.localizedDescription.UTF8String);
    exit(5);
  }
  return state;
}

void matmul(id<MTLCommandBuffer> cb, id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c, int n) {
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pipeline(cb.device)];
  [enc setBuffer:a offset:0 atIndex:0];
  [enc setBuffer:b offset:0 atIndex:1];
  [enc setBuffer:c offset:0 atIndex:2];
  [enc setBytes:&n length:sizeof n atIndex:3];
  [enc dispatchThreads:MTLSizeMake(n, n, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [enc endEncoding];
}

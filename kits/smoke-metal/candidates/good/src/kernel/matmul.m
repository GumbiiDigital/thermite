#import "matmul.h"

/*
 * Threadgroup tiling: each threadgroup loads 64x8 and 8x64 tiles of A and B into threadgroup
 * memory, and each of its 256 threads keeps a 4x4 block of C in registers, so it does 16
 * multiply-adds for every 8 values it reads. This is several times faster than the naive kernel.
 */
static NSString *const SOURCE = @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "#define BM 64\n"
    "#define BN 64\n"
    "#define BK 8\n"
    "#define TM 4\n"
    "#define TN 4\n"
    "kernel void matmul_kernel(device const float *a [[buffer(0)]],\n"
    "                          device const float *b [[buffer(1)]],\n"
    "                          device float *c [[buffer(2)]],\n"
    "                          constant int &n [[buffer(3)]],\n"
    "                          uint2 group [[threadgroup_position_in_grid]],\n"
    "                          uint lid [[thread_index_in_threadgroup]]) {\n"
    "  threadgroup float as[BK][BM];\n"
    "  threadgroup float bs[BK][BN];\n"
    "  int tx = lid % (BN / TN), ty = lid / (BN / TN);\n"
    "  int row0 = group.y * BM, col0 = group.x * BN;\n"
    "  float acc[TM][TN] = {};\n"
    "  for (int t = 0; t < n; t += BK) {\n"
    "    for (int i = lid; i < BM * BK; i += 256) {\n"
    "      int m = i / BK, k = i % BK;\n"
    "      as[k][m] = (row0 + m < n && t + k < n) ? a[(row0 + m) * n + t + k] : 0.0f;\n"
    "    }\n"
    "    for (int i = lid; i < BK * BN; i += 256) {\n"
    "      int k = i / BN, j = i % BN;\n"
    "      bs[k][j] = (t + k < n && col0 + j < n) ? b[(t + k) * n + col0 + j] : 0.0f;\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (int k = 0; k < BK; k++) {\n"
    "      float ra[TM], rb[TN];\n"
    "      for (int i = 0; i < TM; i++) ra[i] = as[k][ty * TM + i];\n"
    "      for (int j = 0; j < TN; j++) rb[j] = bs[k][tx * TN + j];\n"
    "      for (int i = 0; i < TM; i++)\n"
    "        for (int j = 0; j < TN; j++) acc[i][j] += ra[i] * rb[j];\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  }\n"
    "  for (int i = 0; i < TM; i++)\n"
    "    for (int j = 0; j < TN; j++) {\n"
    "      int r = row0 + ty * TM + i, col = col0 + tx * TN + j;\n"
    "      if (r < n && col < n) c[r * n + col] = acc[i][j];\n"
    "    }\n"
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
  [enc dispatchThreadgroups:MTLSizeMake((n + 63) / 64, (n + 63) / 64, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [enc endEncoding];
}

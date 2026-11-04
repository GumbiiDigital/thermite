/*
 * Dispatch probe: counts calls to the reference. Apple's linker has no --wrap, so build.sh
 * compiles the reference under another name (-Dmatmul_ref=thermite_real_matmul_ref), and this
 * file is the matmul_ref every other file links to: it counts, then calls the real one.
 */
#import "matmul.h"

unsigned long long thermite_ref_calls = 0;

void thermite_real_matmul_ref(id<MTLCommandBuffer> cb, id<MTLBuffer> a, id<MTLBuffer> b,
                              id<MTLBuffer> c, int n);

void matmul_ref(id<MTLCommandBuffer> cb, id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c,
                int n) {
  thermite_ref_calls++;
  thermite_real_matmul_ref(cb, a, b, c, n);
}

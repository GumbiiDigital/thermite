/*
 * Metal smoke kit benchmark driver (from the kit; candidates cannot change it). It runs the
 * fixed work for one cell on the Mac's GPU, then writes the result with the dispatch counter.
 * The judge times the whole process from outside, so the reps queue on the GPU without a wait
 * between them: the time is the kernel's, not the CPU's round trip after every rep.
 *
 * usage: bench <params-file> <out-dir>
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#import "matmul.h"

extern unsigned long long thermite_ref_calls;

static long param(const char *path, const char *key) {
  FILE *f = fopen(path, "r");
  if (!f) {
    perror(path);
    exit(2);
  }
  char line[256];
  size_t klen = strlen(key);
  while (fgets(line, sizeof line, f)) {
    if (strncmp(line, key, klen) == 0 && line[klen] == '=') {
      fclose(f);
      return strtol(line + klen + 1, NULL, 10);
    }
  }
  fclose(f);
  fprintf(stderr, "missing parameter %s\n", key);
  exit(2);
}

/* The same sequence as the CUDA and CPU smoke kits, so check.mjs is theirs. */
static unsigned int lcg_state;
static float lcg(void) {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return (float)(lcg_state >> 8) / 16777216.0f - 0.5f;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 3) {
      fprintf(stderr, "usage: bench <params-file> <out-dir>\n");
      return 2;
    }
    const int n = (int)param(argv[1], "n");
    const long reps = param(argv[1], "reps");
    lcg_state = (unsigned int)param(argv[1], "seed");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "no Metal device\n");
      return 5;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSUInteger bytes = sizeof(float) * (NSUInteger)n * n;
    id<MTLBuffer> a = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> c = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!queue || !a || !b || !c) return 3;
    float *ha = (float *)a.contents, *hb = (float *)b.contents;
    for (int i = 0; i < n * n; i++) ha[i] = lcg();
    for (int i = 0; i < n * n; i++) hb[i] = lcg();
    /* Every rep changes one input element, so no rep's result can be reused for another. The
       new values come from the same sequence, in the same order, as before: A, B, then one per
       rep. They are made here, before any rep runs, into a buffer of the kit's own, and each
       rep's command buffer starts with a copy of its value into A on the GPU. The CPU never
       writes A while reps run. */
    id<MTLBuffer> values = [device newBufferWithLength:sizeof(float) * (NSUInteger)(reps > 0 ? reps : 1)
                                               options:MTLResourceStorageModeShared];
    if (!values) return 3;
    float *hv = (float *)values.contents;
    for (long r = 0; r < reps; r++) hv[r] = lcg();
    /* The reps are committed without waiting. Up to WINDOW command buffers are in flight (the
       queue may hold fewer): rep r waits for rep r - WINDOW first, which caps the memory they
       hold without draining the GPU. Every command buffer is checked for an error once it has
       completed. */
    enum { WINDOW = 256 };
    id<MTLCommandBuffer> ring[WINDOW] = {nil};
    for (long r = 0; r < reps + WINDOW; r++) {
      @autoreleasepool {
        /* Rep r - WINDOW's slot: wait for it and check it. The last WINDOW turns only wait and
           check, so every rep has completed and been checked before the result is read. */
        id<MTLCommandBuffer> done = ring[r % WINDOW];
        if (done) {
          [done waitUntilCompleted];
          if (done.error) {
            fprintf(stderr, "rep %ld: %s\n", r - WINDOW, done.error.localizedDescription.UTF8String);
            return 5;
          }
          ring[r % WINDOW] = nil;
        }
        if (r >= reps) continue;
        long at = (r * 7919) % ((long)n * n);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:values
                 sourceOffset:sizeof(float) * (NSUInteger)r
                     toBuffer:a
            destinationOffset:sizeof(float) * (NSUInteger)at
                         size:sizeof(float)];
        [blit endEncoding];
        matmul(cb, a, b, c, n);
        [cb commit];
        ring[r % WINDOW] = cb;
      }
    }

    char path[4096];
    snprintf(path, sizeof path, "%s/c.bin", argv[2]);
    FILE *out = fopen(path, "wb");
    if (!out || fwrite(c.contents, sizeof(float), (size_t)n * n, out) != (size_t)n * n) return 4;
    fclose(out);
    snprintf(path, sizeof path, "%s/probe.txt", argv[2]);
    FILE *probe = fopen(path, "w");
    if (!probe) return 4;
    fprintf(probe, "ref_calls=%llu\n", thermite_ref_calls);
    fclose(probe);
  }
  return 0;
}

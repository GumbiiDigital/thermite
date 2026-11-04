/*
 * Metal smoke kit benchmark driver (from the kit; candidates cannot change it). It runs the
 * fixed work for one cell on the Mac's GPU, then writes the result with the dispatch counter.
 * The judge times the whole process from outside.
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
       buffers are shared with the GPU, and each rep waits for its work, so the write is safe. */
    for (long r = 0; r < reps; r++) {
      @autoreleasepool {
        long at = (r * 7919) % ((long)n * n);
        ha[at] = lcg();
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        matmul(cb, a, b, c, n);
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
          fprintf(stderr, "rep %ld: %s\n", r, cb.error.localizedDescription.UTF8String);
          return 5;
        }
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

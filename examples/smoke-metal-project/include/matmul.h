#ifndef MATMUL_H
#define MATMUL_H

#import <Metal/Metal.h>

/*
 * C = A * B for n x n row-major float matrices in shared-storage buffers on cb's device.
 * Encodes the work into the command buffer and returns: the caller commits and waits.
 */
void matmul(id<MTLCommandBuffer> cb, id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c, int n);

/* Reference implementation. Candidates may not call it from matmul(). */
void matmul_ref(id<MTLCommandBuffer> cb, id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c, int n);

#endif

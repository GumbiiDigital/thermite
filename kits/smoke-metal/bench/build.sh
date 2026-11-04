#!/bin/sh
# Builds the Metal smoke benchmark with clang from Xcode or the Command Line Tools. The kernels
# are Metal Shading Language strings that each program compiles at run time, so no Metal
# toolchain is needed.
#
# The reference is compiled under another name (-Dmatmul_ref=thermite_real_matmul_ref), and the
# kit's probe.m is the matmul_ref the rest links to: it counts the calls (Apple's linker has no
# --wrap).
#
# The judge builds with an empty environment (only PATH). On some Macs clang's default SDK does
# not link there: a Command Line Tools SDK newer than the linker fails with "tapi error:
# malformed file ... unknown architecture". So it tries, in order, no -isysroot, the SDK xcrun
# names, and Xcode's own SDK, and keeps the first that links.
set -u
mkdir -p out

try() {
  clang -O2 -fobjc-arc -Iinclude "$@" -Dmatmul_ref=thermite_real_matmul_ref \
    -c -o out/matmul_ref.o src/ref/matmul_ref.m &&
    clang -O2 -fobjc-arc -Iinclude "$@" -framework Metal -framework Foundation \
      -o out/bench .kit/bench/main.m .kit/bench/probe.m src/kernel/*.m out/matmul_ref.o
}

if try 2>out/build-1.log; then exit 0; fi
sdk=$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)
if [ -n "$sdk" ] && try -isysroot "$sdk" 2>out/build-2.log; then exit 0; fi
xcode="$(xcode-select -p 2>/dev/null || true)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
if [ -d "$xcode" ] && try -isysroot "$xcode" 2>out/build-3.log; then exit 0; fi
echo "the Metal smoke benchmark did not build with any SDK:" >&2
cat out/build-*.log >&2
exit 1

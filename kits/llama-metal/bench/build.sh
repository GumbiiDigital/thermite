#!/bin/sh
# Builds llama.cpp's llama and ggml libraries for Metal, statically, and links the kit's driver
# (.kit/bench/main.cpp) against them into one program, out/bench. Run in the candidate's source
# tree. The Metal kernels are embedded as source (GGML_METAL_EMBED_LIBRARY) and compiled when the
# program starts, so no Metal toolchain is needed: Xcode's clang or the Command Line Tools' does.
#
# The judge builds with an empty environment (only PATH). On some Macs clang's default SDK does
# not link there: a Command Line Tools SDK newer than the linker fails with "tapi error:
# malformed file ... unknown architecture". So it tries, in order, no SDK, the SDK xcrun names,
# and Xcode's own SDK, and keeps the first that builds.
set -u
mkdir -p out

if command -v ninja >/dev/null 2>&1; then gen=Ninja; else gen="Unix Makefiles"; fi
jobs=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)

try() {
  dir=$1
  shift
  sysroot=""
  [ $# -gt 0 ] && sysroot=$1
  rm -rf "$dir"
  set -- -S . -B "$dir" -G "$gen" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DGGML_CCACHE=OFF
  [ -n "$sysroot" ] && set -- "$@" -DCMAKE_OSX_SYSROOT="$sysroot"
  cmake "$@" &&
    cmake --build "$dir" --target llama -j "$jobs" &&
    libs=$(find "$dir" -name '*.a' | sort) &&
    [ -n "$libs" ] &&
    # shellcheck disable=SC2086 # the libraries' paths have no spaces: they are under $dir.
    clang++ -std=c++17 -O2 ${sysroot:+-isysroot "$sysroot"} -Iinclude -Iggml/include \
      -o out/bench .kit/bench/main.cpp $libs \
      -framework Metal -framework MetalKit -framework Foundation -framework Accelerate
}

if try out/cmake >out/build-1.log 2>&1; then exit 0; fi
sdk=$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)
if [ -n "$sdk" ] && try out/cmake "$sdk" >out/build-2.log 2>&1; then exit 0; fi
xcode="$(xcode-select -p 2>/dev/null || true)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
if [ -d "$xcode" ] && try out/cmake "$xcode" >out/build-3.log 2>&1; then exit 0; fi
echo "llama.cpp did not build with any SDK:" >&2
tail -n 40 out/build-*.log >&2
exit 1

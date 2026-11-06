#!/bin/sh
# Builds llama.cpp's llama and ggml libraries for Metal, statically, and links the kit's driver
# (.kit/bench/main.cpp) against them into one program, out/bench. Run in the candidate's source
# tree.
#
# The Metal kernels are compiled here, at build time, into out/default.metallib, which ggml loads
# from beside the program. They are not embedded as source (GGML_METAL_EMBED_LIBRARY is off): the
# fence closes Metal's shader cache to every role, so source compiled at start-up costs every
# timed run about 8 s, which hides kernel wins. So the build needs the Metal compiler (metal and
# metallib). It looks on PATH, then asks xcrun, then takes the newest Metal Toolchain that macOS
# keeps mounted (Xcode's Metal Toolchain component); with none, it stops.
#
# ggml compiles its source at start-up with GGML_METAL_HAS_BF16, and with GGML_METAL_HAS_TENSOR
# on GPUs that use the tensor API (M5 and later); the library here gets the same, and the driver
# turns ggml's tensor API off when the library has no tensor kernels (KIT_METAL_TENSOR), so the
# kernels ggml asks for always match the library's.
#
# The judge builds with an empty environment (only PATH, and TMPDIR inside the build). On some
# Macs clang's default SDK does not link there: a Command Line Tools SDK newer than the linker
# fails with "tapi error: malformed file ... unknown architecture". So it tries, in order, no
# SDK, the SDK xcrun names, and Xcode's own SDK, and keeps the first that builds.
set -u
mkdir -p out

# The Metal compiler: PATH, then xcrun, then the newest mounted Metal Toolchain.
metal=$(command -v metal 2>/dev/null || true)
metallib=$(command -v metallib 2>/dev/null || true)
if [ -z "$metal" ] || [ -z "$metallib" ]; then
  metal=$(xcrun -sdk macosx -f metal 2>/dev/null || true)
  metallib=$(xcrun -sdk macosx -f metallib 2>/dev/null || true)
fi
if [ -z "$metal" ] || [ -z "$metallib" ]; then
  bin=$(ls -d /var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin 2>/dev/null |
    sort -V | tail -n 1)
  if [ -n "$bin" ] && [ -x "$bin/metal" ] && [ -x "$bin/metallib" ]; then
    metal=$bin/metal
    metallib=$bin/metallib
  fi
fi
if [ -z "$metal" ] || [ -z "$metallib" ]; then
  echo "the Metal compiler is missing: install Xcode's Metal Toolchain, or put metal and metallib on PATH outside the home folder, which the fence hides" >&2
  exit 1
fi

# The kernels, as ggml's CMakeLists.txt compiles them (-O3), with the macros ggml's start-up
# compile would set on this Mac's GPU. The module cache stays in out/: the judge's build has no
# home folder and no shared temporary folder.
rm -rf out/metal-modules out/ggml-metal.air out/default.metallib
case $(/usr/sbin/sysctl -n machdep.cpu.brand_string 2>/dev/null) in
  *M5* | *M6* | *A19* | *A20*) tensor=1 ;;
  *) tensor=0 ;;
esac
kernels() {
  "$metal" -O3 -fmodules-cache-path="$PWD/out/metal-modules" -DGGML_METAL_HAS_BF16=1 "$@" \
    -Iggml/src -Iggml/src/ggml-metal -c ggml/src/ggml-metal/ggml-metal.metal -o out/ggml-metal.air &&
    "$metallib" out/ggml-metal.air -o out/default.metallib
}
if [ "$tensor" = 1 ] && ! kernels -DGGML_METAL_HAS_TENSOR=1 >out/metal.log 2>&1; then
  echo "the Metal kernels did not compile with the tensor API this GPU has:" >&2
  tail -n 40 out/metal.log >&2
  exit 1
fi
if [ "$tensor" = 0 ] && ! kernels >>out/metal.log 2>&1; then
  echo "the Metal kernels did not compile:" >&2
  tail -n 40 out/metal.log >&2
  exit 1
fi
rm -f out/ggml-metal.air

if command -v ninja >/dev/null 2>&1; then gen=Ninja; else gen="Unix Makefiles"; fi
jobs=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)

try() {
  dir=$1
  shift
  sysroot=""
  [ $# -gt 0 ] && sysroot=$1
  rm -rf "$dir"
  set -- -S . -B "$dir" -G "$gen" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=OFF -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DGGML_CCACHE=OFF
  [ -n "$sysroot" ] && set -- "$@" -DCMAKE_OSX_SYSROOT="$sysroot"
  # Only the llama target: ggml-metal-lib would call `xcrun -sdk macosx metal`, which fails
  # where only the Command Line Tools are installed. The kernels are compiled above.
  cmake "$@" &&
    cmake --build "$dir" --target llama -j "$jobs" &&
    libs=$(find "$dir" -name '*.a' | sort) &&
    [ -n "$libs" ] &&
    # shellcheck disable=SC2086 # the libraries' paths have no spaces: they are under $dir.
    clang++ -std=c++17 -O2 ${sysroot:+-isysroot "$sysroot"} -Iinclude -Iggml/include \
      -DKIT_METAL_TENSOR="$tensor" -o out/bench .kit/bench/main.cpp $libs \
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

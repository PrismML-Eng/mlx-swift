# MLX-Swift with 1-Bit Quantization Support

This fork of [mlx-swift](https://github.com/ml-explore/mlx-swift) adds native 1-bit weight quantization, enabling any MLX 1-bit model to run efficiently on iPhone and iPad at significantly reduced memory footprints.

## Quick Start

### Option A: Swift Package Dependency

Add this fork as a dependency (replacing the standard mlx-swift URL):

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/PrismML-Eng/mlx-swift.git", branch: "prism"),
]
```

No changes are needed in your model loading code. If a model's `config.json` specifies `"bits": 1`, the 1-bit kernels are dispatched automatically through `QuantizedLinear`.

### Option B: Build from Source

```bash
git clone https://github.com/PrismML-Eng/mlx-swift.git
cd mlx-swift
git checkout prism
git submodule update --init
```

Regenerate Metal shaders and build with Xcode 27:

```bash
./tools/update-mlx.sh
swift build
```

With Xcode 26, use Xcode's package build so that it also builds the Metal library:

```sh
xcodebuild build -scheme mlx-swift-Package -destination 'platform=macOS'
```

## Quantization Format

| Property | Value |
|----------|-------|
| Bits per weight | 1 (packed into uint32, 32 values per word) |
| Group size | 32, 64, or 128 |
| Per-group parameters | fp16 `scale` + fp16 `bias` |
| Effective storage | ~1.1–1.5 bits/weight (depending on group size) |
| Dequantization | `w = scale * bit + bias` |

Models use SafeTensors format with `config.json` containing:
```json
{"quantization": {"bits": 1, "group_size": 128}}
```

## Related Repositories

- [PrismML-Eng/mlx](https://github.com/PrismML-Eng/mlx/tree/prism) — MLX C++ core with 1-bit kernel support
- [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) — Upstream mlx-swift
- [ml-explore/mlx](https://github.com/ml-explore/mlx) — Upstream MLX framework

## Upstream 0.32.2 compatibility

The `prism` development line integrates the Swift 0.32.2 release while retaining
the fork's low-bit kernels and signed Hadamard layers. Its pinned MLX core includes
upstream MLX 0.32.2 plus subsequent fork changes; this is not an unmodified upstream
0.32.2 core.

The older `v0.31.6_prism` branch is unchanged. Consumers pinned to that branch do
not receive development-line fixes automatically. Update the package revision and
rebuild the Metal library together; do not reuse a library from the older runtime.

M5 desktop GPUs use the Neural Accelerator paths supported by the pinned core.
`PrismNAXRegressionTests` compares dense and low-bit matmuls at the historical M5
failure shapes, and head-dimension-256 attention, against FP32 CPU references. The
tests cover FP16 and BF16 inputs and 1-, 2-, and 4-bit affine weights. Additional
cases cover FP32/FP16 split-K tiles (including partial tiles) and short non-causal
head-dimension-128 attention.

Run the focused checks on a Metal-capable Mac with Xcode 27:

```sh
swift test -c release -Xswiftc -DDEBUG \
  --filter 'Prism|Hadamard|StreamTests|DeviceTests|SaveTests'
```

The debug define enables the upstream test suite's wired-memory testing hooks.
With Xcode 26, `swift test` can fail with `Failed to load the default metallib`.
Use the Xcode package scheme, which builds the Metal library, instead:

```sh
xcodebuild test -scheme mlx-swift-Package -configuration Debug \
  -destination 'platform=macOS' \
  -only-testing:MLXTests/PrismNAXRegressionTests \
  -only-testing:MLXTests/PrismHadamardTests \
  -only-testing:MLXTests/HadamardLayerTests \
  -only-testing:MLXTests/HadamardFusedInputTests \
  -only-testing:MLXTests/PrismReleaseQuantizationTests \
  -only-testing:MLXTests/PrismSymmetricQuantizationTests \
  -only-testing:MLXTests/StreamTests \
  -only-testing:MLXTests/SaveTests
```

For consumer migration, review the upstream release's task-local stream/device
semantics and changed defaults for `tensordot`, `nanToNum`, and `linspace`.

### Build and CI scope

The fork's current GitHub workflow gates lint and macOS build/test jobs on the
upstream repository name. Linux build jobs depend on lint, so they are also
skipped here. Passing CodeQL checks do not establish that these builds or tests
ran. The Linux container changes require separate build validation.

Use SwiftPM or the Xcode package scheme for the fork kernels. The existing CMake
build fetches upstream MLX and MLX-C instead of the fork's pinned submodules and
does not provide the fork kernels. The MLX-C submodule itself currently uses the
`bri-prism/mlx-c` fork; cloning with submodules requires access to that repository.

---

## Appendix

### What Changed

#### MLX C++ core (submodule: `Source/Cmlx/mlx`)

The mlx submodule points to [PrismML-Eng/mlx](https://github.com/PrismML-Eng/mlx/tree/prism) which adds:

- **Validation** (`ops.cpp`): Accepts `bits=1` in quantize/dequantize operations
- **Metal kernels** (`quantized.h`, `quantized_nax.h`): 1-bit `load_vector`, `qdot`, `qouter`, `dequantize` using bit extraction and `select()` intrinsics
- **Kernel instantiation** (`quantized.metal`): `instantiate_quantized_groups(1)` for all group sizes
- **CPU backend** (`cpu/quantized.cpp`): 1-bit dequantization path

#### MLX-Swift level

- **`tools/update-mlx.sh`**: Regenerates Metal sources and headers from the pinned core and C bindings
- **`.gitmodules`**: Points mlx submodule to the 1-bit fork
- **`Source/Cmlx/mlx-generated/`**: Regenerated Metal shaders with 1-bit support

## Hadamard-folded Bonsai 2 checkpoints

`MLXNN` provides `SignedBlockHadamard`, `HadamardQuantizedLinear`, and
`HadamardQuantizedEmbedding`. The transform applies explicit signs across the
full input width, accumulates in FP32, and restores the activation dtype.
Embedding lookup applies the inverse transform; tied output projection applies
the forward transform.

A `prism_hadamard_qwen35` checkpoint also needs model-loader support in
`mlx-swift-lm`. The runtime dependency alone does not register its model type or
install its transformed layers. The loader must validate the checkpoint's module
manifest and install the declared layers before strict weight loading. Loading
these tensors as ordinary quantized layers produces incorrect outputs.

The integration covers the published Qwen3.5-compatible Bonsai 2 27B artifact:
2-bit affine weights, group size 128, FP16 activations, and explicit signed
Hadamard blocks. It does not convert or rewrite the checkpoint.

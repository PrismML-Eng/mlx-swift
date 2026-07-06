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

Apply patches (needed until upstreamed):

```bash
cd Source/Cmlx/mlx
git apply ../../../patches/mlx-quantized-dispatch-1bit.patch
cd ../../..

cd Source/Cmlx/mlx-c
git apply ../../../patches/mlx-c-global-scale-nullopt.patch
cd ../../..
```

Regenerate Metal shaders and build:

```bash
./tools/update-mlx.sh
swift build
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

---

## Appendix

### What Changed

#### MLX C++ core (submodule: `Source/Cmlx/mlx`)

The mlx submodule points to [PrismML-Eng/mlx](https://github.com/PrismML-Eng/mlx/tree/prism) which adds:

- **Validation** (`ops.cpp`): Accepts `bits=1` in quantize/dequantize operations
- **Metal kernels** (`quantized.h`, `quantized_nax.h`): 1-bit `load_vector`, `qdot`, `qouter`, `dequantize` using bit extraction and `select()` intrinsics
- **Kernel instantiation** (`quantized.metal`): `instantiate_quantized_groups(1)` for all group sizes
- **CPU backend** (`cpu/quantized.cpp`): 1-bit dequantization path

#### Patches (applied on top of submodules)

| Patch | File | Change |
|-------|------|--------|
| `mlx-quantized-dispatch-1bit.patch` | `mlx/backend/metal/quantized.cpp` | Guards fast-path kernel dispatch for 1-bit on mobile Metal GPUs. (2 lines) |
| `mlx-c-global-scale-nullopt.patch` | `mlx-c/mlx/c/ops.cpp` | Passes `std::nullopt` for new `global_scale` parameters in `quantize`, `dequantize`, and `qqmm` C bindings. (4 lines) |

#### MLX-Swift level

- **`tools/update-mlx.sh`**: Added `steel_conv_3d` build target (required after merging upstream mlx changes)
- **`.gitmodules`**: Points mlx submodule to the 1-bit fork
- **`Source/Cmlx/mlx-generated/`**: Regenerated Metal shaders with 1-bit support

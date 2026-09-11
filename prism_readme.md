# Hadamard quantization in MLXNN

`MLXNN` provides `SignedBlockHadamard`, `PrismHadamardConfiguration`,
`HadamardQuantizedLinear`, `HadamardQuantizedEmbedding`, and `HadamardGDNLayout`.
The layers extend the official release API without changing existing layer behavior.

Decode extracted version-1 `prism.hadamard.*` metadata and construct a layer from
MLX affine packed arrays without requantizing:

```swift
import Foundation
import MLX
import MLXNN

let configuration = try JSONDecoder().decode(
    PrismHadamardConfiguration.self, from: metadataJSON)
let transform = try configuration.transform(forWidth: inputWidth)
let projection = try HadamardQuantizedLinear(
    weight: packedWeight, scales: scales, biases: quantizationBiases,
    groupSize: 128, bits: 2, transform: transform)
let output = projection(activations)
```

`metadataJSON` is the extracted metadata object, not a GGUF file. `packedWeight`
is a uint32 MLX matrix. Raw PQ2/PTQ1/Q2 GGUF blocks require conversion before they
can be supplied here. The initializer validates packed dimensions and scale/bias
shapes. Malformed signs, unsupported transform conventions, and unknown sign
widths throw errors during configuration loading.

`HadamardQuantizedEmbedding` accepts the same packed-weight arguments and applies
the inverse transform after lookup. Its `asLinear` method applies the forward
transform for tied output weights. Both layer types conform to `Quantized`, so
`quantizeSingle` does not quantize them again. Transform signs are immutable
configuration rather than trainable parameters. Save the metadata alongside
parameter arrays when saving a checkpoint.

When the model's GDN produces tiled values and `configuration.gdnVGrouped` is
true, pass `HadamardGDNLayout(width:keyHeads:valueHeads:)` as the linear layer's
`gdnLayout` argument for the GDN output projection. It permutes tiled values to
grouped values before applying signs and Hadamard. Do not enable it when GDN
already produces grouped values or on unrelated projections. Tensor-to-module
mapping and the upstream GDN layout remain the model loader's responsibility.

### Validation

Metal tests on M5 Pro with Xcode 26.6 cover block sizes 512/1024/2048/4096 at ten
compatible model-width pairs, row counts 1/8/64, and 1/2-bit affine matmul through
the production layer. They compare against independent scalar butterflies.
Additional tests cover embedding lookup, tied projection, FP32/FP16/BF16,
grouped GDN ordering, invalid metadata and packed shapes, and existing NAX and
quantization regressions. This is runtime/layer validation; real-model generation
and raw GGUF loading are not established by these tests.

## Release dependency

Use the fork's `v0.31.6_prism` branch for the release-based runtime:

```swift
.package(url: "https://github.com/PrismML-Eng/mlx-swift.git", branch: "v0.31.6_prism")
```

A version requirement such as `from: "0.31.6"` resolves version tags rather than
this branch. Keep the resolved dependency file to record tested revisions.
Ordinary builds use the checked-in generated sources and require no manual
patch application. For a source checkout, initialize submodules recursively.

The core pin includes the release backport of the generation-18 NAX guard. Its
source guard agrees with the checked-in flattened header, so regeneration
preserves the guard.

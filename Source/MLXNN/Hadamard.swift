import Foundation
import MLX

/// Invalid transform metadata or incompatible packed weights.
public enum HadamardError: Error {
    case invalidBlockSize
    case invalidSigns
    case incompatibleShape
    case unsupportedContract
    case missingSignWidth(Int)
}

/// A normalized block Walsh-Hadamard transform with explicit input signs.
///
/// Forward evaluation computes `(x * signs) H`; inverse evaluation computes
/// `(x H) * signs`. Signs cover the entire final dimension, not just one block.
public struct SignedBlockHadamard {
    public let blockSize: Int
    public let width: Int
    private let signs: MLXArray
    private let signValues: [Float]

    public init(blockSize: Int, signs: [Float]) throws {
        guard blockSize > 0, blockSize <= 8192,
            blockSize & (blockSize - 1) == 0
        else { throw HadamardError.invalidBlockSize }
        guard !signs.isEmpty, signs.count % blockSize == 0,
            signs.allSatisfy({ $0 == -1 || $0 == 1 })
        else { throw HadamardError.invalidSigns }
        self.blockSize = blockSize
        self.width = signs.count
        self.signs = MLXArray(signs)
        self.signValues = signs
    }

    /// Check serialized signs against the independently decoded metadata.
    public func matches(signs values: [Float]) -> Bool { values == signValues }

    /// True when both transforms compute the same function. Transforms decoded
    /// for one width share one sign buffer, so the common case is O(1).
    public func isIdentical(to other: SignedBlockHadamard) -> Bool {
        blockSize == other.blockSize && width == other.width
            && (signs === other.signs || signValues == other.signValues)
    }

    /// An override for the one-launch forward transform (the library's own
    /// `FusedHadamardKernel` is the default). It receives the activation, the sign vector, the block
    /// size, whether the activation already carries the signs, an optional
    /// GDN layout to gather through, and the output dtype. It must compute
    /// exactly what the op chain computes (FP32 sign multiply, the FP32
    /// block transform with the stock kernel's butterfly order and scale,
    /// then one cast to the output dtype). Returns nil to decline.
    public typealias FusedTransform = (
        _ x: MLXArray, _ signs: MLXArray, _ blockSize: Int, _ preSigned: Bool,
        _ gdnLayout: HadamardGDNLayout?, _ outputDType: DType
    ) -> MLXArray?
    nonisolated(unsafe) public static var fusedTransform: FusedTransform?

    /// Transform activations before multiplication by folded weights.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        forward(x, gdnLayout: nil, outputDType: x.dtype)
    }

    /// The forward transform of `x` (optionally gathered through a GDN
    /// layout first), returned in `outputDType`. Same values as
    /// `callAsFunction(layout(x)).asType(outputDType)`.
    public func forward(_ x: MLXArray, gdnLayout: HadamardGDNLayout?, outputDType: DType)
        -> MLXArray
    {
        validate(x)
        if let fused = Self.fusedTransform,
            let y = fused(x, signs, blockSize, false, gdnLayout, outputDType)
        {
            return y
        }
        if let y = FusedHadamardKernel.transform(
            x, signs: signs, blockSize: blockSize, preSigned: false, gdnLayout: gdnLayout,
            outputDType: outputDType)
        {
            return y
        }
        let laidOut = gdnLayout.map { $0(x) } ?? x
        let rotated = hadamardTransform(
            (laidOut.asType(.float32) * signs).reshaped([-1, blockSize])
        ).reshaped(x.shape).asType(x.dtype)
        return rotated.dtype == outputDType ? rotated : rotated.asType(outputDType)
    }

    /// `applyPreSigned((silu(gate) * up) * signVector)` with the SwiGLU product
    /// and the sign flip formed inside the fused rotation's read: the compiled
    /// `(silu(gate.asType(.float32)) * up.asType(.float32)) * signs` chain, each
    /// product rounded to FP32 in that order, MLX's `Sigmoid` verbatim. Gate
    /// and up may be FP32 or FP16 (widened exactly). Nil when it does not apply.
    public func rotatedSwiGLU(
        gate: MLXArray, up: MLXArray, outputDType: DType = .float32
    ) -> MLXArray? {
        guard FusedInputHadamardKernel.swigluEnabled, blockSize == 1024, width % 1024 == 0,
            gate.dtype == up.dtype, gate.dtype == .float32 || gate.dtype == .float16,
            gate.shape == up.shape, gate.ndim > 0, gate.dim(-1) == width
        else { return nil }
        return FusedInputHadamardKernel.gated(
            gate, up, signs: signs, width: width, mode: 1, outputDType: outputDType)
    }

    /// `self(x * sigmoid(gate))`, the attention output gate, fused the same way.
    public func rotatedSigmoidGate(
        _ x: MLXArray, gate: MLXArray, outputDType: DType = .float32
    ) -> MLXArray? {
        guard FusedInputHadamardKernel.gateEnabled, blockSize == 1024, width % 1024 == 0,
            x.dtype == .float32, gate.dtype == .float32, x.shape == gate.shape,
            x.ndim > 0, x.dim(-1) == width
        else { return nil }
        return FusedInputHadamardKernel.gated(
            x, gate, signs: signs, width: width, mode: 2, outputDType: outputDType)
    }

    /// `self(layout((silu(z) * rmsNorm(x, weight, eps)).reshaped(width)))` for
    /// the GDN output: per-head RMSNorm exactly as MLX's `rms_single_row` over a
    /// 128-wide head (32 lanes x 4 reads, `simd_sum`, `precise::rsqrt`,
    /// `w * (x * inv)`), the compiled `silu(z) * normed` tail, the value-head
    /// permutation, the signs and the transform, in one kernel.
    public func rotatedGatedRMSNorm(
        _ x: MLXArray, gate z: MLXArray, weight: MLXArray, eps: Float,
        layout: HadamardGDNLayout?, outputDType: DType = .float32
    ) -> MLXArray? {
        guard x.ndim == 4 else { return nil }
        let headDim = x.dim(-1)
        let valueHeads = x.dim(-2)
        // No layout: heads stay in place (repeats 1). Grouped layout: the
        // value-head permutation is folded into the reads.
        let repeats = layout.map { $0.valueHeads / $0.keyHeads } ?? 1
        let keyHeads = layout?.keyHeads ?? valueHeads
        guard FusedInputHadamardKernel.gatedNormEnabled, blockSize == 1024,
            width == valueHeads * headDim, width % 1024 == 0,
            headDim == 128, 1024 % headDim == 0,
            layout == nil || (layout!.width == width && layout!.valueHeads == valueHeads),
            repeats * keyHeads == valueHeads, x.shape == z.shape, x.dtype == .float32,
            z.dtype == .float32, weight.dtype == .float32, weight.ndim == 1,
            weight.dim(0) == headDim
        else { return nil }
        return FusedInputHadamardKernel.gatedRMSNorm(
            x, z, weight: weight, eps: eps, signs: signs,
            repeats: repeats, keyHeads: keyHeads, headDim: headDim,
            outputDType: outputDType)
    }

    /// The sign vector as an array, for a caller that folds the sign flip into
    /// an elementwise op it already runs on the activation. Read only.
    public var signVector: MLXArray { signs }

    /// The forward transform of an activation that already carries the signs
    /// (`x * signVector`, in FP32). Identical to `callAsFunction` on the
    /// unsigned activation; the multiply has simply been done by the caller.
    public func applyPreSigned(_ signed: MLXArray) -> MLXArray {
        applyPreSigned(signed, outputDType: signed.dtype)
    }

    /// `applyPreSigned` returned in `outputDType` (the cast folded into the
    /// transform when a fused implementation is installed).
    public func applyPreSigned(_ signed: MLXArray, outputDType: DType) -> MLXArray {
        validate(signed)
        if let fused = Self.fusedTransform,
            let y = fused(signed, signs, blockSize, true, nil, outputDType)
        {
            return y
        }
        if let y = FusedHadamardKernel.transform(
            signed, signs: signs, blockSize: blockSize, preSigned: true, gdnLayout: nil,
            outputDType: outputDType)
        {
            return y
        }
        let rotated = hadamardTransform(signed.asType(.float32).reshaped([-1, blockSize]))
            .reshaped(signed.shape).asType(signed.dtype)
        return rotated.dtype == outputDType ? rotated : rotated.asType(outputDType)
    }

    /// Recover the original basis after looking up folded embedding rows.
    public func inverse(_ x: MLXArray) -> MLXArray {
        validate(x)
        return
            (hadamardTransform(x.asType(.float32).reshaped([-1, blockSize])).reshaped(x.shape)
            * signs).asType(x.dtype)
    }

    private func validate(_ x: MLXArray) {
        precondition(x.ndim > 0 && x.dim(-1) == width, "Hadamard input width mismatch")
        precondition(
            [DType.float32, .float16, .bfloat16].contains(x.dtype),
            "Hadamard input must have a real floating-point dtype")
    }
}

/// The version-1 `prism.hadamard.*` metadata exported as a JSON object.
///
/// This decodes extracted metadata, not GGUF bytes. Tensor names remain in the
/// source namespace. A model loader must map them to its modules and honor
/// `gdnVGrouped` when arranging GDN values before an output projection.
public struct PrismHadamardConfiguration: Decodable {
    public let blockSize: Int
    public let weightNames: [String]
    public let inverseWeightNames: [String]
    public let gdnVGrouped: Bool
    private let transforms: [Int: SignedBlockHadamard]

    private enum CodingKeys: String, CodingKey {
        case version = "prism.hadamard.version"
        case blockSize = "prism.hadamard.block_size"
        case transform = "prism.hadamard.transform"
        case axis = "prism.hadamard.axis"
        case signMode = "prism.hadamard.sign_mode"
        case weightNames = "prism.hadamard.weight_names"
        case inverseWeightNames = "prism.hadamard.inverse_weight_names"
        case signWidths = "prism.hadamard.sign_widths"
        case signValues = "prism.hadamard.sign_values"
        case gdnVGrouped = "prism.hadamard.gdn_v_grouped"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard try c.decode(Int.self, forKey: .version) == 1,
            try c.decode(String.self, forKey: .transform) == "normalized-sylvester-walsh-hadamard",
            try c.decode(String.self, forKey: .axis) == "input-last-dimension",
            try c.decode(String.self, forKey: .signMode) == "explicit"
        else { throw HadamardError.unsupportedContract }
        blockSize = try c.decode(Int.self, forKey: .blockSize)
        weightNames = try c.decode([String].self, forKey: .weightNames)
        inverseWeightNames = try c.decodeIfPresent([String].self, forKey: .inverseWeightNames) ?? []
        gdnVGrouped = try c.decodeIfPresent(Bool.self, forKey: .gdnVGrouped) ?? false
        guard Set(weightNames).count == weightNames.count,
            Set(inverseWeightNames).count == inverseWeightNames.count,
            Set(weightNames).isDisjoint(with: inverseWeightNames),
            (weightNames + inverseWeightNames).allSatisfy({ !$0.isEmpty })
        else { throw HadamardError.unsupportedContract }
        let widths = try c.decode([Int].self, forKey: .signWidths)
        let values = try c.decode([Float].self, forKey: .signValues)
        guard !widths.isEmpty, Set(widths).count == widths.count else {
            throw HadamardError.invalidSigns
        }
        var offset = 0
        var transforms = [Int: SignedBlockHadamard]()
        for width in widths {
            guard width > 0, width <= values.count - offset else {
                throw HadamardError.invalidSigns
            }
            transforms[width] = try SignedBlockHadamard(
                blockSize: blockSize, signs: Array(values[offset ..< offset + width]))
            offset += width
        }
        guard offset == values.count else { throw HadamardError.invalidSigns }
        self.transforms = transforms
    }

    public func transform(forWidth width: Int) throws -> SignedBlockHadamard {
        guard let transform = transforms[width] else {
            throw HadamardError.missingSignWidth(width)
        }
        return transform
    }
}

private func validateHadamardWeights(
    _ weight: MLXArray, scales: MLXArray, biases: MLXArray?,
    groupSize: Int, bits: Int, transform: SignedBlockHadamard
) throws {
    guard [1, 2, 3, 4, 5, 6, 8].contains(bits), [32, 64, 128].contains(groupSize),
        weight.ndim == 2, weight.dtype == .uint32,
        weight.dim(0) > 0, weight.dim(1) == transform.width / 32 * bits,
        transform.width % 32 == 0, transform.width % groupSize == 0,
        scales.shape == [weight.dim(0), transform.width / groupSize],
        [DType.float32, .float16, .bfloat16].contains(scales.dtype),
        biases == nil || (biases!.shape == scales.shape && biases!.dtype == scales.dtype)
    else { throw HadamardError.incompatibleShape }
}

/// Reorders tiled GDN values into the grouped order of folded output weights.
///
/// The final axis changes from `[repeat, keyHead, headDimension]` to
/// `[keyHead, repeat, headDimension]` before signs and Hadamard are applied.
/// Use only when the upstream GDN produces tiled values; already grouped values
/// must not be permuted again.
public struct HadamardGDNLayout {
    public let width: Int
    public let keyHeads: Int
    public let valueHeads: Int

    public init(width: Int, keyHeads: Int, valueHeads: Int) throws {
        guard width > 0, keyHeads > 0, valueHeads > 0,
            valueHeads % keyHeads == 0, width % valueHeads == 0
        else { throw HadamardError.incompatibleShape }
        self.width = width
        self.keyHeads = keyHeads
        self.valueHeads = valueHeads
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim > 0 && x.dim(-1) == width, "GDN input width mismatch")
        let repeats = valueHeads / keyHeads
        if repeats == 1 { return x }
        return x.reshaped([-1, repeats, keyHeads, width / valueHeads])
            .transposed(0, 2, 1, 3).reshaped(x.shape)
    }
}

/// Input-independent operands for the matrix-regime route of a packed
/// projection: its FP32-widened constants and, for the first of a group of
/// siblings, their stacked operand. A plain class, never a Module or an
/// MLXArray, so reflecting the owning layer cannot add any of these to the
/// parameter tree. Nothing here depends on a request; it is keyed on the
/// layer's own frozen constants.
private final class HadamardMatrixRouteOperands {
    private let lock = NSLock()
    /// Sibling projections fused along their output axis; see
    /// `HadamardFusedSiblings`. Owned by the first sibling's operands.
    var fusedSiblings: HadamardFusedSiblings?

    func clear() {
        lock.withLock { fusedSiblings = nil }
    }

    /// The stacked operand for exactly these siblings, built on first use and
    /// rebuilt only when a sibling or its weight object changes.
    func fusedSiblings(for siblings: [HadamardQuantizedLinear]) -> HadamardFusedSiblings {
        lock.withLock {
            if let fusedSiblings, fusedSiblings.matches(siblings) {
                return fusedSiblings
            }
            let built = HadamardFusedSiblings(siblings)
            fusedSiblings = built
            return built
        }
    }
}

/// Several packed projections that read one rotated activation, stacked along
/// their output axis into one packed operand: the rows of `weight`, `scales`
/// and `biases` are the siblings' rows in order, byte for byte. One matmul
/// then replaces one per sibling, and the wide result is split back. The
/// stacking copies packed rows; it does not unpack, requantize or re-scale
/// anything. A plain class for the same reason as the route operands.
private final class HadamardFusedSiblings {
    let siblingIDs: [ObjectIdentifier]
    let weightSources: [MLXArray]
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    /// Cumulative output boundaries; the last entry is the total width.
    let boundaries: [Int]
    let operands = HadamardMatrixRouteOperands()

    init(_ siblings: [HadamardQuantizedLinear]) {
        siblingIDs = siblings.map { ObjectIdentifier($0) }
        weightSources = siblings.map(\.weight)
        weight = concatenated(siblings.map(\.weight), axis: 0)
        scales = concatenated(siblings.map(\.scales), axis: 0)
        if siblings.allSatisfy({ $0.biases != nil }) {
            biases = concatenated(siblings.map { $0.biases! }, axis: 0)
        } else {
            biases = nil
        }
        var edges = [Int]()
        var total = 0
        for sibling in siblings {
            total += sibling.weight.dim(0)
            edges.append(total)
        }
        boundaries = edges
    }

    /// True when this stack was built from exactly these layers holding
    /// exactly these weight objects.
    func matches(_ siblings: [HadamardQuantizedLinear]) -> Bool {
        guard siblings.count == siblingIDs.count else { return false }
        for (index, sibling) in siblings.enumerated() {
            guard ObjectIdentifier(sibling) == siblingIDs[index],
                sibling.weight === weightSources[index]
            else { return false }
        }
        return true
    }
}

/// Affine packed linear weights with a signed Hadamard input transform.
///
/// Pass weights already folded and packed in MLX format. The initializer does
/// not requantize them and does not accept raw GGUF quantization blocks.
public final class HadamardQuantizedLinear: QuantizedLinear {
    public let transform: SignedBlockHadamard
    public let gdnLayout: HadamardGDNLayout?
    public init(
        weight: MLXArray, bias: MLXArray? = nil, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, transform: SignedBlockHadamard,
        gdnLayout: HadamardGDNLayout? = nil
    ) throws {
        try validateHadamardWeights(
            weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, transform: transform)
        guard
            bias == nil
                || (bias!.shape == [weight.dim(0)]
                    && [DType.float32, .float16, .bfloat16].contains(bias!.dtype))
        else {
            throw HadamardError.incompatibleShape
        }
        guard gdnLayout == nil || gdnLayout!.width == transform.width else {
            throw HadamardError.incompatibleShape
        }
        self.gdnLayout = gdnLayout
        self.transform = transform
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits)
        freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        applyRotated(rotate(x))
    }

    /// The input transform alone: GDN layout, signs, Hadamard, dtype restore.
    public func rotate(_ x: MLXArray) -> MLXArray {
        transform.forward(x, gdnLayout: gdnLayout, outputDType: x.dtype)
    }

    /// `rotate` returned in `outputDType` (one cast folded into the transform
    /// when a fused implementation is installed).
    public func rotate(_ x: MLXArray, outputDType: DType) -> MLXArray {
        transform.forward(x, gdnLayout: gdnLayout, outputDType: outputDType)
    }

    /// The packed matmul on an input already passed through `rotate`.
    public func applyRotated(_ rotated: MLXArray) -> MLXArray {
        if let routed = matrixRegimeForward(rotated) {
            return routed
        }
        return super.callAsFunction(rotated)
    }

    // MARK: - Matrix-regime route

    /// On unless explicitly disabled. See `matrixRegimeForward`.
    private static let matrixRouteEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment[
            "MLX_HADAMARD_MATRIX_ROUTE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The dtype the packed matmul reads its rotated activation in on the
    /// matrix-regime route. FP16 is the published Prism runtime's own choice
    /// for this pack (its packed constants are FP16, so nothing is widened);
    /// `MLX_HADAMARD_PACKED_INPUT=float32` keeps the FP32 read and the
    /// FP32-widened constants instead.
    private static let matrixRouteInputDType: DType = {
        let value = ProcessInfo.processInfo.environment[
            "MLX_HADAMARD_PACKED_INPUT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch value {
        case "float32", "fp32", "f32": return .float32
        default: return .float16
        }
    }()

    /// The core's vector-versus-matrix threshold for this pack's shapes on the
    /// M5 generation: fewer rows than this take the scalar vector kernel.
    private static let matrixRegimeMinimumRows = 13
    /// A projection at least this wide is a vocabulary head. Its products are
    /// logits that an argmax reads directly, so it keeps the FP32 read (TF32
    /// tensor products, FP32 logits) and the cached widened constants; only
    /// the tower's projections take the FP16 read.
    private static let vocabularyHeadMinimumRows = 65536
    /// The core splits K whenever the 32x32 tile count is at most this.
    private static let splitKTileCeiling = 256
    /// The split-K tensor body takes FP16 input for one 16-row half of its
    /// 32-row tile; a split-K projection over more rows keeps the FP32 read.
    private static let splitKHalfRowLimit = 16

    /// On unless explicitly disabled: a narrow (split-K) projection at a
    /// verify width reads its rotated activation in the route dtype too.
    private static let narrowHalfRead: Bool = {
        let value = ProcessInfo.processInfo.environment[
            "MLX_HADAMARD_NARROW_HALF"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private let matrixRoute = HadamardMatrixRouteOperands()

    @discardableResult
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        matrixRoute.clear()
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    /// The packed matmul for a multi-row input, routed onto the M5 matrix
    /// kernels for every projection of the tower.
    ///
    /// The core dispatch (`QuantizedMatmul::eval_gpu`) sends fewer than 13
    /// rows to the scalar `qmv_wide` kernel, which pays a device load and an
    /// FMA per weight per row, and 13 or more rows to the tensor `qmm_t_nax`
    /// kernel or, for a projection with at most 256 32x32 tiles, to the
    /// split-K kernel whose body is on the tensor unit for FP32 input. This
    /// route keeps the same weights and changes only what the kernels see:
    ///
    /// - rows below the threshold are zero-padded up to it, so the core takes
    ///   the matrix path (the padded rows are dropped from the result);
    /// - a tower projection reads its rotated activation in
    ///   `matrixRouteInputDType` (FP16: the packed FP16 constants are used as
    ///   stored, the FP16 result is widened back so every consumer sees the
    ///   dtype it saw before), a wide one on `qmm_t_nax` and a narrow one on
    ///   the split-K tensor body up to 16 rows; a narrow one over more rows and
    ///   the vocabulary head keep the FP32 read with the cached widened constants;
    /// - a BF16 input (the drafter reading the shared head) is widened to
    ///   FP32 exactly, as the core would, and reuses the cached widened
    ///   constants instead of casting them on every call.
    ///
    /// The tensor kernels already round an FP32 input to a 10-bit mantissa,
    /// which is FP16's mantissa, so the FP16 read changes the range and the
    /// output rounding rather than the product precision. The token gate
    /// prices that. Returns nil when the route does not apply.
    private func matrixRegimeForward(_ x: MLXArray, widenOutput: Bool = true) -> MLXArray? {
        guard Self.routeApplies(to: self), x.dtype == .float32 || x.dtype == .bfloat16,
            x.ndim >= 2
        else { return nil }
        let k = x.dim(-1)
        guard x.size / k >= 2, k % 64 == 0, k % groupSize == 0 else { return nil }
        return Self.matrixRoutedMatmul(
            x, weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode, operands: matrixRoute,
            widenOutput: widenOutput)
    }

    /// `callAsFunction` for a consumer that promotes dtypes itself, such as
    /// the residual add: when the route applies, the FP16 product is returned
    /// as is instead of being widened first. The consumer's promotion widens
    /// the same values exactly, so the arithmetic is unchanged and one cast
    /// dispatch per call is saved.
    public func forwardUnwidened(_ x: MLXArray) -> MLXArray {
        let rotated = rotate(x)
        return matrixRegimeForward(rotated, widenOutput: false) ?? applyRotated(rotated)
    }

    /// The projection of an activation that already carries the transform's
    /// signs (see `SignedBlockHadamard.applyPreSigned`), optionally leaving the
    /// FP16 product unwidened. Not for a layer with a GDN layout.
    public func forwardPreSigned(_ signed: MLXArray, widenOutput: Bool = true) -> MLXArray {
        precondition(gdnLayout == nil, "pre-signed forward needs an ungrouped layout")
        let rotated = transform.applyPreSigned(signed)
        if !widenOutput, let routed = matrixRegimeForward(rotated, widenOutput: false) {
            return routed
        }
        return applyRotated(rotated)
    }

    /// The dtype a fused-input rotation feeding this layer alone should store:
    /// the dtype the matrix route reads for `rows` rows of an FP32 activation
    /// (so the route never casts it again). Nil when the route does not apply.
    private func fusedInputStoreDType(rows: Int) -> DType? {
        let k = transform.width
        guard Self.routeApplies(to: self), rows >= 2, k % 64 == 0, k % groupSize == 0
        else { return nil }
        return Self.routeInputDType(rows: rows, n: weight.dim(0), sourceDType: .float32)
    }

    /// The routed matmul of a fused-input rotation stored in the route dtype
    /// on behalf of an FP32 activation (the FP32 contract is kept for the
    /// output widening).
    private func fusedInputForward(_ rotated: MLXArray, widenOutput: Bool) -> MLXArray {
        Self.matrixRoutedMatmul(
            rotated, weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode, operands: matrixRoute,
            widenOutput: widenOutput, sourceDType: .float32)
    }

    /// `forwardPreSigned((silu(gate) * up) * signs)` with the SwiGLU product,
    /// the signs, the transform and the route dtype's rounding in one kernel
    /// (ercumentyildirim, `ade7529`). Nil when it does not apply.
    public func applyAfterSwiGLU(gate: MLXArray, up: MLXArray, widenOutput: Bool = true)
        -> MLXArray?
    {
        guard gdnLayout == nil,
            let store = fusedInputStoreDType(rows: gate.size / max(transform.width, 1)),
            let rotated = transform.rotatedSwiGLU(gate: gate, up: up, outputDType: store)
        else { return nil }
        return fusedInputForward(rotated, widenOutput: widenOutput)
    }

    /// `self(x * sigmoid(gate))` with the gate product fused into the rotation.
    public func applyAfterSigmoidGate(_ x: MLXArray, gate: MLXArray, widenOutput: Bool = true)
        -> MLXArray?
    {
        guard gdnLayout == nil,
            let store = fusedInputStoreDType(rows: x.size / max(transform.width, 1)),
            let rotated = transform.rotatedSigmoidGate(x, gate: gate, outputDType: store)
        else { return nil }
        return fusedInputForward(rotated, widenOutput: widenOutput)
    }

    /// The GDN output projection of `silu(z) * rmsNorm(x, weight, eps)` with the
    /// norm, the gate, the value layout and the rotation in one kernel.
    public func applyAfterGatedRMSNorm(
        _ x: MLXArray, gate z: MLXArray, weight: MLXArray, eps: Float, widenOutput: Bool = true
    ) -> MLXArray? {
        guard let store = fusedInputStoreDType(rows: x.size / max(transform.width, 1)),
            let rotated = transform.rotatedGatedRMSNorm(
                x, gate: z, weight: weight, eps: eps, layout: gdnLayout, outputDType: store)
        else { return nil }
        return fusedInputForward(rotated, widenOutput: widenOutput)
    }

    /// The representation the route handles: the pack's 2-bit affine layout
    /// with FP16 constants, no linear bias, and a 64-aligned output width.
    private static func routeApplies(to layer: HadamardQuantizedLinear) -> Bool {
        matrixRouteEnabled && layer.mode == .affine && layer.bits == 2 && layer.bias == nil
            && layer.scales.dtype == .float16 && layer.weight.dim(0) % 64 == 0
    }

    /// The routed matmul over `x` (any leading shape, `[..., K]`) with the
    /// given packed operand; returns `[..., N]` in `x.dtype`.
    /// The activation dtype the route reads for a projection of `n` output
    /// rows at `rows` input rows whose activation was originally `sourceDType`.
    static func routeInputDType(rows: Int, n: Int, sourceDType: DType) -> DType {
        let paddedRows = max(rows, matrixRegimeMinimumRows)
        let nTiles = (n + 31) / 32
        let mTiles = (paddedRows + 31) / 32
        let narrow = nTiles * mTiles <= splitKTileCeiling
        // A narrow (split-K) projection reads FP16 up to one 16-row half of
        // its tile (fkiene `0fa9a35`); above that it keeps the FP32 read.
        let narrowFloat32 =
            narrow && !(narrowHalfRead && paddedRows <= splitKHalfRowLimit)
        return (narrowFloat32 || n >= vocabularyHeadMinimumRows || sourceDType == .bfloat16)
            ? .float32 : matrixRouteInputDType
    }

    private static func matrixRoutedMatmul(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode,
        operands: HadamardMatrixRouteOperands, widenOutput: Bool = true,
        sourceDType: DType? = nil
    ) -> MLXArray {
        let k = x.dim(-1)
        let rows = x.size / k
        let n = weight.dim(0)
        let paddedRows = max(rows, matrixRegimeMinimumRows)
        // The dtype the caller's activation had before the rotation; a fused
        // rotation may already have produced `x` in the route dtype.
        let sourceDType = sourceDType ?? x.dtype

        // The core splits K for a projection with at most 256 32x32 tiles and
        // runs the split-K body; that body is on the tensor unit for FP32 input
        // and, for one 16-row half, for FP16 input, so a narrow projection (o,
        // out, down on this pack) reads FP16 at a verify width and FP32 above
        // it. A wide one takes `qmm_t_nax` in the route dtype. BF16
        // activations (the drafter's head input) widen to FP32 exactly, as
        // the core's own promotion would, and a vocabulary head keeps FP32.
        let inputDType = Self.routeInputDType(
            rows: rows, n: n, sourceDType: sourceDType)
        let routeScales: MLXArray
        let routeBiases: MLXArray?
        if inputDType == .float32 {
            // The FP32 read widens the packed FP16 constants, as the core's
            // own promotion would.
            routeScales = scales.asType(.float32)
            routeBiases = biases.map { $0.asType(.float32) }
        } else {
            routeScales = scales
            routeBiases = biases
        }

        var input = x.reshaped(rows, k)
        if inputDType != x.dtype {
            input = input.asType(inputDType)
        }
        if paddedRows > rows {
            input = concatenated(
                [input, MLXArray.zeros([paddedRows - rows, k], dtype: inputDType)], axis: 0)
        }

        var output = quantizedMM(
            input, weight, scales: routeScales, biases: routeBiases,
            transpose: true, groupSize: groupSize, bits: bits, mode: mode)
        if paddedRows > rows {
            output = output[0 ..< rows]
        }
        // The core promotes a BF16 activation with FP16 constants to FP32 and
        // returns FP32; the widened result therefore matches what the plain
        // operator would have returned for either input dtype.
        let plainOutputDType: DType = sourceDType == .bfloat16 ? .float32 : sourceDType
        if widenOutput, output.dtype != plainOutputDType {
            output = output.asType(plainOutputDType)
        }
        return output.reshaped(Array(x.shape.dropLast()) + [n])
    }

    // MARK: - Fused siblings

    /// On unless explicitly disabled. See `fusedSiblingsForward`.
    private static let siblingFusionEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment[
            "MLX_HADAMARD_FUSE_SIBLINGS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// One routed matmul for several siblings that read the same rotated
    /// activation, over their packed rows stacked along the output axis, split
    /// back into one result per sibling.
    ///
    /// Two effects at a verify width. The stack has a wide output, so the core
    /// gives it the 32-row tensor kernel where a narrow sibling on its own
    /// (k, v, z) would have been split-K or, on this route, a 64-row gather
    /// tile. And one dispatch replaces one per sibling for the matmul and for
    /// each surrounding cast, with the rotated activation read once.
    ///
    /// Returns nil when the siblings do not all fit the route (a caller then
    /// applies each one to the rotated activation as before).
    /// True when `fusedSiblingsForward` will take these siblings for an
    /// activation of this shape (the route applies to every sibling).
    fileprivate func fusedSiblingsApply(
        _ siblings: [HadamardQuantizedLinear], rows: Int, k: Int
    ) -> Bool {
        guard Self.siblingFusionEnabled, siblings.count >= 2, rows >= 2, k % 64 == 0
        else { return false }
        for sibling in siblings {
            guard Self.routeApplies(to: sibling), sibling.groupSize == groupSize,
                sibling.weight.dim(1) == weight.dim(1), k % sibling.groupSize == 0
            else { return false }
        }
        return true
    }

    /// The dtype the stacked sibling matmul reads at `rows` input rows.
    fileprivate func fusedSiblingsInputDType(
        _ siblings: [HadamardQuantizedLinear], rows: Int, sourceDType: DType
    ) -> DType {
        let n = siblings.reduce(0) { $0 + $1.weight.dim(0) }
        return Self.routeInputDType(rows: rows, n: n, sourceDType: sourceDType)
    }

    fileprivate func fusedSiblingsForward(
        _ rotated: MLXArray, siblings: [HadamardQuantizedLinear], widenOutput: Bool = true,
        sourceDType: DType = .float32
    ) -> [MLXArray]? {
        guard rotated.ndim >= 2, rotated.dtype == .float32 || rotated.dtype == .float16
        else { return nil }
        let k = rotated.dim(-1)
        guard fusedSiblingsApply(siblings, rows: rotated.size / k, k: k) else { return nil }
        let fused = matrixRoute.fusedSiblings(for: siblings)
        let wide = Self.matrixRoutedMatmul(
            rotated, weight: fused.weight, scales: fused.scales, biases: fused.biases,
            groupSize: groupSize, bits: bits, mode: mode, operands: fused.operands,
            widenOutput: widenOutput, sourceDType: sourceDType)
        return MLX.split(wide, indices: Array(fused.boundaries.dropLast()), axis: -1)
    }

    /// True when `rotate` is the same function on both layers, so one rotated
    /// activation can feed both packed matmuls with bit-identical results.
    public func sharesInputTransform(with other: HadamardQuantizedLinear) -> Bool {
        gdnLayout == nil && other.gdnLayout == nil
            && transform.isIdentical(to: other.transform)
    }
}

/// Applies each packed Hadamard projection to the same activation, rotating it
/// once. Every projection reads the identical rotated array it would have
/// computed itself, so outputs are bit-identical to calling each one. Returns
/// nil when any projection is not packed or uses a different transform.
public func sharedHadamardProjections(
    _ x: MLXArray, _ projections: [Linear], widenOutput: Bool = true
) -> [MLXArray]? {
    guard let first = projections.first as? HadamardQuantizedLinear else { return nil }
    var packed = [HadamardQuantizedLinear]()
    packed.reserveCapacity(projections.count)
    for projection in projections {
        guard let layer = projection as? HadamardQuantizedLinear,
            layer.sharesInputTransform(with: first)
        else { return nil }
        packed.append(layer)
    }
    // When the siblings will run as one routed stack, rotate straight into the
    // dtype that stack reads: the fused transform folds the cast into its
    // single launch, and the route then has nothing left to cast.
    let k = x.dim(-1)
    if x.dtype == .float32, first.fusedSiblingsApply(packed, rows: x.size / k, k: k) {
        let routeDType = first.fusedSiblingsInputDType(
            packed, rows: x.size / k, sourceDType: x.dtype)
        let rotated = first.rotate(x, outputDType: routeDType)
        if let fused = first.fusedSiblingsForward(
            rotated, siblings: packed, widenOutput: widenOutput, sourceDType: x.dtype)
        {
            return fused
        }
        let plain = routeDType == x.dtype ? rotated : rotated.asType(x.dtype)
        return packed.map { $0.applyRotated(plain) }
    }
    let rotated = first.rotate(x)
    if let fused = first.fusedSiblingsForward(
        rotated, siblings: packed, widenOutput: widenOutput)
    {
        return fused
    }
    return packed.map { $0.applyRotated(rotated) }
}

/// The packed projections that share one transform, when every one of them is
/// packed with that same transform and none has a GDN layout; nil otherwise.
public func sharedHadamardSiblings(_ projections: [Linear]) -> [HadamardQuantizedLinear]? {
    guard let first = projections.first as? HadamardQuantizedLinear else { return nil }
    var packed = [HadamardQuantizedLinear]()
    packed.reserveCapacity(projections.count)
    for projection in projections {
        guard let layer = projection as? HadamardQuantizedLinear,
            layer.sharesInputTransform(with: first)
        else { return nil }
        packed.append(layer)
    }
    return packed
}

/// `sharedHadamardProjections` for an activation that already carries the
/// shared transform's signs (see `SignedBlockHadamard.applyPreSigned`): the
/// rotation skips its sign multiply, and every sibling reads the rotated
/// array `sharedHadamardProjections` would have formed from the unsigned
/// activation. Nil when the siblings do not share one ungrouped transform.
public func sharedHadamardProjectionsPreSigned(
    _ signed: MLXArray, _ siblings: [HadamardQuantizedLinear], widenOutput: Bool = true
) -> [MLXArray]? {
    guard let first = siblings.first,
        siblings.allSatisfy({ $0.sharesInputTransform(with: first) })
    else { return nil }
    // As in `sharedHadamardProjections`: when the siblings run as one routed
    // stack, rotate straight into the dtype the stack reads.
    let k = signed.dim(-1)
    if signed.dtype == .float32, first.fusedSiblingsApply(siblings, rows: signed.size / k, k: k) {
        let routeDType = first.fusedSiblingsInputDType(
            siblings, rows: signed.size / k, sourceDType: signed.dtype)
        let rotated = first.transform.applyPreSigned(signed, outputDType: routeDType)
        if let fused = first.fusedSiblingsForward(
            rotated, siblings: siblings, widenOutput: widenOutput, sourceDType: signed.dtype)
        {
            return fused
        }
        let plain = routeDType == signed.dtype ? rotated : rotated.asType(signed.dtype)
        return siblings.map { $0.applyRotated(plain) }
    }
    let rotated = first.transform.applyPreSigned(signed)
    if let fused = first.fusedSiblingsForward(
        rotated, siblings: siblings, widenOutput: widenOutput)
    {
        return fused
    }
    return siblings.map { $0.applyRotated(rotated) }
}

/// Packed folded embeddings with an inverse transform after lookup.
///
/// `asLinear` applies the forward transform, allowing the same packed weights
/// to serve as a tied output projection without unfolding the full vocabulary.
public final class HadamardQuantizedEmbedding: Embedding, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode = .affine
    public let scales: MLXArray
    public let biases: MLXArray?
    public let transform: SignedBlockHadamard
    /// When set, dequantized rows are cast to this dtype before the inverse
    /// transform (the published pack restores FP16 embedding activations).
    public let outputDType: DType?

    public override var shape: (Int, Int) { (weight.dim(0), transform.width) }

    public init(
        weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, transform: SignedBlockHadamard, outputDType: DType? = nil
    ) throws {
        try validateHadamardWeights(
            weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, transform: transform)
        self.groupSize = groupSize
        self.bits = bits
        self.scales = scales
        self.biases = biases
        self.transform = transform
        self.outputDType = outputDType
        super.init(weight: weight)
        freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let indices = x.flattened()
        let rows = dequantized(
            weight[indices], scales: scales[indices],
            biases: biases.map { $0[indices] }, groupSize: groupSize, bits: bits)
        return transform.inverse(rows.asType(outputDType ?? rows.dtype)).reshaped(
            x.shape + [transform.width])
    }

    public override func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            transform(x), weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits)
    }
}

/// ercumentyildirim's (`ade7529`) fused-INPUT rotations: the SwiGLU product,
/// the attention output gate, or the GDN output's per-head RMSNorm and gated
/// tail, formed in the read of MLX's `hadamard_n<float, 1024, 16, 4>` with the
/// signs, and the result stored once in the dtype the packed matmul reads.
/// Every product is the composed op chain's, rounded to FP32 in the same
/// order, so the stored values equal the chain's FP32 rotation cast to that
/// dtype. (The plain rotation is `FusedHadamardKernel`; these kernels have
/// their own names.)
enum FusedInputHadamardKernel {
    private static func flag(_ name: String) -> Bool {
        let value = ProcessInfo.processInfo.environment[name]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }
    /// Per-path switches (all default on; `MLX_HADAMARD_FUSED_INPUT=0` turns all off).
    static let plainEnabled = enabled && flag("MLX_HADAMARD_FUSED_PLAIN")
    static let swigluEnabled = enabled && flag("MLX_HADAMARD_FUSED_SWIGLU")
    static let gateEnabled = enabled && flag("MLX_HADAMARD_FUSED_GATE")
    static let gatedNormEnabled = enabled && flag("MLX_HADAMARD_FUSED_GNORM")

    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLX_HADAMARD_FUSED_INPUT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static func applies(blockSize: Int, width: Int, dtype: DType) -> Bool {
        plainEnabled && blockSize == 1024 && width % 1024 == 0 && dtype == .float32
    }

    static func gated(
        _ a: MLXArray, _ b: MLXArray, signs: MLXArray, width: Int, mode: Int,
        outputDType: DType = .float32
    ) -> MLXArray {
        gatedKernel(
            [a, b, signs],
            template: [
                ("WIDTH", width), ("MODE", mode), ("InT", a.dtype), ("OutT", outputDType),
            ],
            grid: (64, a.size / 1024, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [a.shape],
            outputDTypes: [outputDType])[0]
    }

    /// MODE 1: `(a * sigmoid(a)) * b` (SwiGLU, inputs FP32 or FP16 widened
    /// exactly). MODE 2: `a * sigmoid(b)`.
    private static let gatedKernel = MLXFast.metalKernel(
        name: "mlxnn_fused_input_gated_hadamard_1024",
        inputNames: ["a", "b", "signs"],
        outputNames: ["out"],
        source: """
            constexpr short NT = 64;
            constexpr uint BLOCKS = WIDTH / 1024;
            short i = short(thread_position_in_grid.x);
            uint blk = thread_position_in_grid.y;
            uint row_base = (blk / BLOCKS) * WIDTH;
            uint col0 = (blk % BLOCKS) * 1024;

            threadgroup float buf[1024];

            MLXNN_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              MLXNN_UNROLL for (short r = 0; r < 4; r++) {
                uint p = col0 + index + r;
                float av = static_cast<float>(a[row_base + p]);
                float bv = static_cast<float>(b[row_base + p]);
                float v;
                if (MODE == 1) {
                  float t = av * mlxnn_sigmoid(av);
                  v = t * bv;
                } else {
                  v = av * mlxnn_sigmoid(bv);
                }
                buf[index + r] = v * signs[p];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float v[16];
            short h = 1;
            MLXNN_UNROLL for (short s = 0; s < 2; s++) {
              short k = i & (h - 1);
              short j = ((i - k) << 4) + k;
              MLXNN_UNROLL for (short r = 0; r < 16; r++) {
                v[r] = buf[j + h * r];
              }
              mlxnn_hadamard_radix<16>(v);
              MLXNN_UNROLL for (short r = 0; r < 16; r++) {
                buf[j + h * r] = v[r];
              }
              h <<= 4;
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            MLXNN_UNROLL for (short t = 0; t < 4; t++) {
              short index = i + t * NT;
              short k = index & (h - 1);
              short j = ((index - k) << 2) + k;
              MLXNN_UNROLL for (short r = 0; r < 4; r++) {
                v[r] = buf[j + h * r];
              }
              mlxnn_hadamard_radix<4>(v);
              MLXNN_UNROLL for (short r = 0; r < 4; r++) {
                buf[j + h * r] = v[r];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            MLXNN_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              MLXNN_UNROLL for (short r = 0; r < 4; r++) {
                out[row_base + col0 + index + r] = static_cast<OutT>(buf[index + r] * 0.03125f);
              }
            }
            """,
        header: """
            #define MLXNN_UNROLL _Pragma("clang loop unroll(full)")

            template <short R>
            METAL_FUNC void mlxnn_hadamard_radix(thread float* x) {
              constexpr short logR = __builtin_ctz(R);
              short h = 1;
              MLXNN_UNROLL for (short s = 0; s < logR; s++) {
                MLXNN_UNROLL for (short i = 0; i < R / 2; i++) {
                  short k = i & (h - 1);
                  short j = ((i - k) << 1) + k;
                  float a = x[j];
                  float b = x[j + h];
                  x[j] = a + b;
                  x[j + h] = a - b;
                }
                h <<= 1;
              }
            }

            // MLX `Sigmoid` (unary_ops.h), verbatim.
            METAL_FUNC float mlxnn_sigmoid(float x) {
              auto y = 1 / (1 + metal::exp(metal::abs(x)));
              return (x < 0) ? y : 1 - y;
            }

            """)
}

extension FusedInputHadamardKernel {
    /// GDN output: per-head RMSNorm (MLX `rms_single_row`, 32 lanes x 4 reads),
    /// `silu(z) * normed`, value-head permutation, signs and the transform.
    static func gatedRMSNorm(
        _ x: MLXArray, _ z: MLXArray, weight: MLXArray, eps: Float, signs: MLXArray,
        repeats: Int, keyHeads: Int, headDim: Int, outputDType: DType = .float32
    ) -> MLXArray {
        let batch = x.dim(0)
        let rows = x.dim(1)
        return gatedRMSNormKernel(
            [x, z, weight, signs, MLXArray(eps)],
            template: [
                ("REPEATS", repeats), ("KEY_HEADS", keyHeads), ("HEAD_DIM", headDim),
                ("OutT", outputDType),
            ],
            grid: (64, x.size / 1024, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[batch, rows, repeats * keyHeads * headDim]],
            outputDTypes: [outputDType])[0]
    }

    private static let gatedRMSNormKernel = MLXFast.metalKernel(
        name: "mlxnn_fused_input_gated_rmsnorm_hadamard_1024",
        inputNames: ["x", "z", "w", "signs", "eps"],
        outputNames: ["out"],
        source: """
            constexpr short NT = 64;
            constexpr uint WIDTH = REPEATS * KEY_HEADS * HEAD_DIM;
            constexpr uint BLOCKS = WIDTH / 1024;
            constexpr uint HEADS_PER_BLOCK = 1024 / HEAD_DIM;
            short i = short(thread_position_in_grid.x);
            uint blk = thread_position_in_grid.y;
            uint row_base = (blk / BLOCKS) * WIDTH;
            uint col0 = (blk % BLOCKS) * 1024;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;

            threadgroup float buf[1024];
            threadgroup float inv_rms[HEADS_PER_BLOCK];

            // Per-head RMS as rms_single_row with 32 threads x 4 reads: lane l
            // sums elements 4l..4l+3 of the head in order, then simd_sum.
            MLXNN_UNROLL for (uint hh = sg; hh < HEADS_PER_BLOCK; hh += 2) {
              uint p0 = col0 + hh * HEAD_DIM;
              uint kh = p0 / (REPEATS * HEAD_DIM);
              uint rep = (p0 % (REPEATS * HEAD_DIM)) / HEAD_DIM;
              uint src_head = rep * KEY_HEADS + kh;
              const device float* xh = x + row_base + src_head * HEAD_DIM + lane * 4;
              float acc = 0;
              float tx[4];
              MLXNN_UNROLL for (int r = 0; r < 4; r++) {
                tx[r] = xh[r];
                acc += tx[r] * tx[r];
              }
              acc = simd_sum(acc);
              if (lane == 0) {
                inv_rms[hh] = metal::precise::rsqrt(acc / HEAD_DIM + eps);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            MLXNN_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              MLXNN_UNROLL for (short r = 0; r < 4; r++) {
                uint p = col0 + index + r;
                uint kh = p / (REPEATS * HEAD_DIM);
                uint rem = p % (REPEATS * HEAD_DIM);
                uint src = ((rem / HEAD_DIM) * KEY_HEADS + kh) * HEAD_DIM + rem % HEAD_DIM;
                float xn = w[src % HEAD_DIM] * (x[row_base + src] * inv_rms[(index + r) / HEAD_DIM]);
                float zv = z[row_base + src];
                float gz = zv * mlxnn_sigmoid(zv);
                float v = gz * xn;
                buf[index + r] = v * signs[p];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float v[16];
            short h = 1;
            MLXNN_UNROLL for (short s = 0; s < 2; s++) {
              short k = i & (h - 1);
              short j = ((i - k) << 4) + k;
              MLXNN_UNROLL for (short r = 0; r < 16; r++) {
                v[r] = buf[j + h * r];
              }
              mlxnn_hadamard_radix<16>(v);
              MLXNN_UNROLL for (short r = 0; r < 16; r++) {
                buf[j + h * r] = v[r];
              }
              h <<= 4;
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            MLXNN_UNROLL for (short t = 0; t < 4; t++) {
              short index = i + t * NT;
              short k = index & (h - 1);
              short j = ((index - k) << 2) + k;
              MLXNN_UNROLL for (short r = 0; r < 4; r++) {
                v[r] = buf[j + h * r];
              }
              mlxnn_hadamard_radix<4>(v);
              MLXNN_UNROLL for (short r = 0; r < 4; r++) {
                buf[j + h * r] = v[r];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            MLXNN_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              MLXNN_UNROLL for (short r = 0; r < 4; r++) {
                out[row_base + col0 + index + r] = static_cast<OutT>(buf[index + r] * 0.03125f);
              }
            }
            """,
        header: """
            #define MLXNN_UNROLL _Pragma("clang loop unroll(full)")

            template <short R>
            METAL_FUNC void mlxnn_hadamard_radix(thread float* x) {
              constexpr short logR = __builtin_ctz(R);
              short h = 1;
              MLXNN_UNROLL for (short s = 0; s < logR; s++) {
                MLXNN_UNROLL for (short i = 0; i < R / 2; i++) {
                  short k = i & (h - 1);
                  short j = ((i - k) << 1) + k;
                  float a = x[j];
                  float b = x[j + h];
                  x[j] = a + b;
                  x[j + h] = a - b;
                }
                h <<= 1;
              }
            }

            // MLX `Sigmoid` (unary_ops.h), verbatim.
            METAL_FUNC float mlxnn_sigmoid(float x) {
              auto y = 1 / (1 + metal::exp(metal::abs(x)));
              return (x < 0) ? y : 1 - y;
            }

            """)
}

/// The one-launch forward transform: the FP32 sign multiply, the 1024-block
/// Walsh-Hadamard butterfly in the stock kernel's stage order (64 threads per
/// block, radix 16-16-4, scale 1/32), the optional GDN value-head gather and
/// one cast to the output dtype, in a single custom kernel. Bit-identical to
/// the op chain. `MLX_HADAMARD_FUSED_KERNEL=0` restores the op chain.
enum FusedHadamardKernel {
    private static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLX_HADAMARD_FUSED_KERNEL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let header = """
        // Thread-local Hadamard butterfly for 2^R values, as in
        // mlx/backend/metal/kernels/hadamard.h (radix_func).
        template <short R>
        inline void mlxnn_hadamard_radix(thread float* x) {
          constexpr short logR = __builtin_ctz(R);
          short h = 1;
          #pragma clang loop unroll(full)
          for (short s = 0; s < logR; s++) {
            #pragma clang loop unroll(full)
            for (short i = 0; i < R / 2; i++) {
              short k = i & (h - 1);
              short j = ((i - k) << 1) + k;
              float a = x[j];
              float b = x[j + h];
              x[j] = a + b;
              x[j + h] = a - b;
            }
            h <<= 1;
          }
        }
        """

    // grid: (64 * blocks, 1, 1), threadgroup (64, 1, 1); one threadgroup per
    // 1024-wide block. Template: InT, OutT, W (row width), BPR (blocks per
    // row), PRESIGNED, GR (GDN repeats, 1 = identity), GKH, GD.
    private static let source = """
        constexpr short N = 1024;
        constexpr short NT = 64;
        const uint blk = threadgroup_position_in_grid.x;
        const short i = short(thread_position_in_threadgroup.x);
        const uint row = blk / uint(BPR);
        const uint bcol = (blk % uint(BPR)) * uint(N);
        const size_t rowbase = size_t(row) * size_t(W);
        threadgroup float buf[N];
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
          const short index = j * 4 * NT + i * 4;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            const uint col = bcol + uint(index + r);
            uint src = col;
            if (GR > 1) {
              const uint d = col % uint(GD);
              const uint hr = col / uint(GD);
              const uint h = hr / uint(GR);
              const uint rr = hr % uint(GR);
              src = (rr * uint(GKH) + h) * uint(GD) + d;
            }
            float v = float(inp[rowbase + src]);
            if (!PRESIGNED) {
              v = v * signs[col];
            }
            buf[index + r] = v;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float x[16];
        short h = 1;
        #pragma clang loop unroll(full)
        for (short s = 0; s < 2; s++) {
          short k = i & (h - 1);
          short j = ((i - k) << 4) + k;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 16; r++) {
            x[r] = buf[j + h * r];
          }
          mlxnn_hadamard_radix<16>(x);
          #pragma clang loop unroll(full)
          for (short r = 0; r < 16; r++) {
            buf[j + h * r] = x[r];
          }
          h <<= 4;
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma clang loop unroll(full)
        for (int t = 0; t < 4; t++) {
          short index = i + t * NT;
          short k = index & (h - 1);
          short j = ((index - k) << 2) + k;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            x[r] = buf[j + h * r];
          }
          mlxnn_hadamard_radix<4>(x);
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            buf[j + h * r] = x[r];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
          const short index = j * 4 * NT + i * 4;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            out[rowbase + bcol + uint(index + r)] = OutT(buf[index + r] * 0.03125f);
          }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "mlxnn_signed_hadamard_1024",
        inputNames: ["inp", "signs"],
        outputNames: ["out"],
        source: source,
        header: header,
        ensureRowContiguous: true)

    /// The transform of `x` (already laid out through `gdnLayout` inside the
    /// kernel), or nil when this kernel does not cover the call.
    static func transform(
        _ x: MLXArray, signs: MLXArray, blockSize: Int, preSigned: Bool,
        gdnLayout: HadamardGDNLayout?, outputDType: DType
    ) -> MLXArray? {
        guard enabled, blockSize == 1024, x.ndim >= 1,
            [DType.float32, .float16, .bfloat16].contains(x.dtype),
            [DType.float32, .float16, .bfloat16].contains(outputDType),
            signs.dtype == .float32
        else { return nil }
        let width = x.dim(-1)
        guard width % 1024 == 0, signs.size == width else { return nil }
        var repeats = 1
        var keyHeads = 1
        var headDim = 1
        if let gdnLayout {
            guard gdnLayout.width == width, gdnLayout.valueHeads % gdnLayout.keyHeads == 0,
                width % gdnLayout.valueHeads == 0
            else { return nil }
            repeats = gdnLayout.valueHeads / gdnLayout.keyHeads
            keyHeads = gdnLayout.keyHeads
            headDim = width / gdnLayout.valueHeads
        }
        let rows = x.size / width
        guard rows > 0 else { return nil }
        let blocksPerRow = width / 1024
        let template: [(String, any KernelTemplateArg)] = [
            ("InT", x.dtype), ("OutT", outputDType), ("W", width), ("BPR", blocksPerRow),
            ("PRESIGNED", preSigned ? 1 : 0), ("GR", repeats), ("GKH", keyHeads), ("GD", headDim),
        ]
        return kernel(
            [x, signs], template: template,
            grid: (64 * rows * blocksPerRow, 1, 1), threadGroup: (64, 1, 1),
            outputShapes: [x.shape], outputDTypes: [outputDType])[0]
    }
}

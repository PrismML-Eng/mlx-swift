import Foundation
import MLX
import MLXNN
import XCTest

// The fused-input rotations read their operands in place, so a gate|up
// half of a stacked product or a `[B, S, heads, headDim]` view produced by a
// transpose reaches the kernel without a contiguous copy. Each case compares
// the fused result on such a strided operand with the unfused chain (the
// elementwise op on a contiguous copy, then the rotation).
class HadamardFusedInputTests: XCTestCase {
    private func transform(width: Int) throws -> SignedBlockHadamard {
        let signs = (0 ..< width).map { ($0 * 17 % 13) < 6 ? Float(-1) : Float(1) }
        return try SignedBlockHadamard(blockSize: 1024, signs: signs)
    }

    private func relativeError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
        let a = actual.asType(.float32)
        let e = expected.asType(.float32)
        return (abs(a - e).max() / (abs(e).max() + 1e-6)).item(Float.self)
    }

    func testSwiGLUReadsStackedHalvesInPlace() throws {
        let width = 5120
        let t = try transform(width: width)
        for dtype in [DType.float32, .float16] {
            for rows in [1, 3, 16] {
                MLXRandom.seed(UInt64(rows))
                let stacked = MLXRandom.normal([1, rows, 2 * width]).asType(dtype)
                let gate = stacked[.ellipsis, 0 ..< width]
                let up = stacked[.ellipsis, width...]
                let fused = try XCTUnwrap(
                    t.rotatedSwiGLU(gate: gate, up: up, outputDType: .float32),
                    "fused SwiGLU declined for \(dtype) rows \(rows)")
                let g = contiguous(gate).asType(.float32)
                let reference = t(silu(g) * contiguous(up).asType(.float32))
                XCTAssertEqual(fused.shape, reference.shape)
                XCTAssertLessThan(relativeError(fused, reference), 1e-5, "\(dtype) rows \(rows)")
            }
        }
    }

    func testSigmoidGateReadsTransposedHeads() throws {
        let heads = 40
        let headDim = 128
        let width = heads * headDim
        let t = try transform(width: width)
        for steps in [1, 3, 16] {
            MLXRandom.seed(UInt64(100 + steps))
            let x = MLXRandom.normal([1, heads, steps, headDim]).transposed(0, 2, 1, 3)
            let gate = MLXRandom.normal([1, heads, steps, headDim]).transposed(0, 2, 1, 3)
            let fused = try XCTUnwrap(
                t.rotatedSigmoidGate(x, gate: gate, outputDType: .float32),
                "fused sigmoid gate declined for steps \(steps)")
            let reference = t(contiguous(x * sigmoid(gate)).reshaped([1, steps, width]))
            XCTAssertEqual(fused.size, reference.size)
            XCTAssertLessThan(
                relativeError(fused.reshaped(reference.shape), reference), 1e-5, "steps \(steps)")
        }
    }

    func testGatedRMSNormReadsTransposedHeads() throws {
        let valueHeads = 48
        let keyHeads = 16
        let headDim = 128
        let width = valueHeads * headDim
        let eps: Float = 1e-6
        let t = try transform(width: width)
        let grouped = try HadamardGDNLayout(
            width: width, keyHeads: keyHeads, valueHeads: valueHeads)
        for layout in [nil, grouped] {
            for steps in [1, 4] {
                MLXRandom.seed(UInt64(200 + steps))
                let x = MLXRandom.normal([1, valueHeads, steps, headDim]).transposed(0, 2, 1, 3)
                let z = MLXRandom.normal([1, valueHeads, steps, headDim]).transposed(0, 2, 1, 3)
                let weight = MLXRandom.uniform(low: 0.5, high: 1.5, [headDim])
                let fused = try XCTUnwrap(
                    t.rotatedGatedRMSNorm(
                        x, gate: z, weight: weight, eps: eps, layout: layout,
                        outputDType: .float32),
                    "fused gated RMSNorm declined for steps \(steps)")
                let xc = contiguous(x)
                let normed = weight * (xc * rsqrt(mean(xc * xc, axis: -1, keepDims: true) + eps))
                var flat = contiguous(silu(z) * normed).reshaped([1, steps, width])
                if let layout { flat = layout(flat) }
                let reference = t(flat)
                XCTAssertEqual(fused.size, reference.size)
                XCTAssertLessThan(
                    relativeError(fused.reshaped(reference.shape), reference), 1e-4,
                    "layout \(layout != nil) steps \(steps)")
            }
        }
    }
}

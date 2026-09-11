import Foundation
import MLX
import MLXNN
import XCTest

class PrismHadamardTests: XCTestCase {
    override class func setUp() {
        setDefaultDevice()
    }

    // Exercise a signed, normalized block transform before the production QMM.
    // This is a runtime contract test, not a model-format compatibility claim.
    func testSignedBlockTransformThenLowBitMatmul() throws {
        let shapes = [
            (2048, 2048), (2048, 6144), (512, 2560), (512, 4096), (512, 9216), (4096, 4096),
            (4096, 12288), (1024, 5120), (1024, 6144), (1024, 17408),
        ]
        for (block, width) in shapes {
            for rows in [1, 8, 64] {
                let values = (0 ..< rows * width).map { Float(($0 * 37) % 127 - 63) / 64 }
                let signs = (0 ..< width).map { ($0 * 17 % 13) < 6 ? Float(-1) : Float(1) }
                var expected = values
                for offset in stride(from: 0, to: expected.count, by: block) {
                    for j in 0 ..< block {
                        expected[offset + j] *= signs[(offset + j) % width]
                    }
                    var step = 1
                    while step < block {
                        for base in stride(from: 0, to: block, by: step * 2) {
                            for j in 0 ..< step {
                                let a = expected[offset + base + j]
                                let b = expected[offset + base + j + step]
                                expected[offset + base + j] = a + b
                                expected[offset + base + j + step] = a - b
                            }
                        }
                        step *= 2
                    }
                    for j in 0 ..< block { expected[offset + j] /= sqrt(Float(block)) }
                }
                let input = MLXArray(values, [rows, width])
                let transform = try SignedBlockHadamard(blockSize: block, signs: signs)
                let transformed = transform(input)
                let restored = transform.inverse(transformed)
                XCTAssertLessThan(abs(restored - input).max().item(Float.self), 1e-5)
                let referenceInput = MLXArray(expected, [rows, width])
                XCTAssertLessThan(abs(transformed - referenceInput).max().item(Float.self), 1e-5)
                let weights = MLXArray(
                    (0 ..< 128 * width).map { Float(($0 * 19) % 131 - 65) / 64 },
                    [128, width])
                for bits in [1, 2] {
                    let (packed, scales, biases) = quantized(weights, groupSize: 128, bits: bits)
                    let layer = try HadamardQuantizedLinear(
                        weight: packed, scales: scales, biases: biases,
                        groupSize: 128, bits: bits, transform: transform)
                    let actual = layer(input)
                    let reference = referenceInput.matmul(
                        dequantized(
                            packed, scales: scales, biases: biases,
                            groupSize: 128, bits: bits
                        ).T)
                    let delta = actual - reference
                    let relative =
                        (delta * delta).sum().sqrt()
                        / (reference * reference).sum().sqrt()
                    XCTAssertTrue(actual.asArray(Float.self).allSatisfy { $0.isFinite })
                    XCTAssertLessThan(
                        relative.item(Float.self), 1e-4,
                        "width=\(width) rows=\(rows) bits=\(bits)")
                }
            }
        }
    }
}

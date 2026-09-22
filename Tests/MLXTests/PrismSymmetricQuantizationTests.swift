import MLX
import XCTest

final class PrismSymmetricQuantizationTests: XCTestCase {
    func testBiasFreeMatmulMatchesExplicitBias() {
        for bits in [1, 2] {
            for rows in [1, 2, 8] {
                for width in [128, 256, 512] {
                    for outputs in [33, 128] {
                        let weights = MLXRandom.normal([outputs, width])
                        let (packed, scales, _) = quantized(weights, groupSize: 128, bits: bits)
                        let x = MLXRandom.normal([rows, width])
                        let bias = -scales * (bits == 1 ? Float(0.5) : Float(1))
                        let expected = quantizedMM(
                            x, packed, scales: scales, biases: bias, groupSize: 128, bits: bits)
                        let actual = quantizedMM(
                            x, packed, scales: scales, biases: nil, groupSize: 128, bits: bits)
                        XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 0.0001)
                    }
                }
            }
        }
    }
}

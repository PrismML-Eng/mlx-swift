import MLX
import XCTest

class PrismReleaseQuantizationTests: XCTestCase {
    override class func setUp() {
        setDefaultDevice()
    }

    func testLowBitMatmulMatchesDequantizedReference() {
        for bits in [1, 2, 4] {
            for groupSize in [32, 64, 128] {
                for rows in [1, 8, 64] {
                    for dtype in [DType.float32, .float16] {
                        let k = 256
                        let n = rows == 1 ? 1024 : 256
                        let input = MLXArray(
                            (0 ..< rows * k).map { Float(($0 * 37) % 127 - 63) / 64 },
                            [rows, k]
                        ).asType(dtype)
                        let weights = MLXArray(
                            (0 ..< n * k).map { Float(($0 * 19) % 131 - 65) / 64 },
                            [n, k]
                        ).asType(dtype)
                        let (packed, scales, biases) = quantized(
                            weights, groupSize: groupSize, bits: bits)
                        XCTAssertEqual(packed.shape, [n, k * bits / 32])
                        XCTAssertNotNil(biases)
                        let unpacked = dequantized(
                            packed, scales: scales, biases: biases,
                            groupSize: groupSize, bits: bits, dtype: .float32)
                        let reference = input.asType(.float32).matmul(unpacked.T)
                        let result = quantizedMM(
                            input, packed, scales: scales, biases: biases,
                            groupSize: groupSize, bits: bits
                        ).asType(.float32)
                        let difference = result - reference
                        let error = (difference * difference).sum().sqrt().item(Float.self)
                        let norm = (reference * reference).sum().sqrt().item(Float.self)
                        XCTAssertTrue(result.asArray(Float.self).allSatisfy { $0.isFinite })
                        // Official 0.31.6 also measures 4.11e-4 for the 4-bit FP32
                        // control on M5 Pro. Keep the Prism low-bit bound stricter.
                        let tolerance: Float =
                            dtype == .float16 ? 0.005 : (bits == 4 ? 0.001 : 0.0001)
                        XCTAssertLessThan(
                            error / max(norm, 1e-6), tolerance,
                            "bits=\(bits) group=\(groupSize) rows=\(rows) dtype=\(dtype)")
                    }
                }
            }
        }
    }
}

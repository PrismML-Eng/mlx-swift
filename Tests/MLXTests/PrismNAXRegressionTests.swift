import MLX
import XCTest

final class PrismNAXRegressionTests: XCTestCase {
    private func assertRelativeError(
        _ actual: MLXArray, _ expected: MLXArray, tolerance: Float,
        context: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        let result = actual.asType(.float32)
        let difference = result - expected
        let error = (difference * difference).sum().sqrt().item(Float.self)
        let norm = (expected * expected).sum().sqrt().item(Float.self)
        XCTAssertTrue(result.asArray(Float.self).allSatisfy { $0.isFinite }, file: file, line: line)
        XCTAssertLessThan(error / max(norm, 1e-6), tolerance, context, file: file, line: line)
    }

    func testDenseMatmulAtM5ThresholdAgainstCPU() {
        // These dimensions crossed the original M5 miscompute threshold.
        for dtype in [DType.float16, .bfloat16] {
            let (input, weight, reference) = Device.withDefaultDevice(.cpu) {
                let input = MLXRandom.normal([16, 512], key: MLXRandom.key(31)).asType(dtype)
                let weight = MLXRandom.normal([512, 16384], key: MLXRandom.key(32)).asType(dtype)
                let reference = matmul(input.asType(.float32), weight.asType(.float32))
                eval(input, weight, reference)
                return (input, weight, reference)
            }
            let result = Device.withDefaultDevice(.gpu) { matmul(input, weight) }
            assertRelativeError(result, reference, tolerance: 0.005, context: "dense \(dtype)")
        }
    }

    func testLowBitMatmulAtM5ThresholdAgainstCPU() {
        for bits in [1, 2, 4] {
            for groupSize in [32, 64, 128] {
                for dtype in [DType.float16, .bfloat16] {
                    let (input, packed, scales, biases, reference) = Device.withDefaultDevice(.cpu)
                    {
                        let input = MLXRandom.normal([64, 512], key: MLXRandom.key(41)).asType(
                            dtype)
                        let weight = MLXRandom.normal([9216, 512], key: MLXRandom.key(42)).asType(
                            dtype)
                        let (packed, scales, biases) = quantized(
                            weight, groupSize: groupSize, bits: bits)
                        let unpacked = dequantized(
                            packed, scales: scales, biases: biases,
                            groupSize: groupSize, bits: bits, dtype: .float32)
                        let reference = matmul(input.asType(.float32), unpacked.T)
                        eval(input, packed, scales, reference)
                        biases?.eval()
                        return (input, packed, scales, biases, reference)
                    }
                    let result = Device.withDefaultDevice(.gpu) {
                        quantizedMM(
                            input, packed, scales: scales, biases: biases,
                            groupSize: groupSize, bits: bits)
                    }
                    assertRelativeError(
                        result, reference, tolerance: 0.01,
                        context: "bits=\(bits) group=\(groupSize) dtype=\(dtype)")
                }
            }
        }
    }

    func testAttentionHead256AgainstCPU() {
        for dtype in [DType.float16, .bfloat16] {
            for length in [64, 1024] {
                let (q, k, v, reference) = Device.withDefaultDevice(.cpu) {
                    let q = MLXRandom.normal([1, 2, length, 256], key: MLXRandom.key(51)).asType(
                        dtype)
                    let k = MLXRandom.normal([1, 1, length, 256], key: MLXRandom.key(52)).asType(
                        dtype)
                    let v = MLXRandom.normal([1, 1, length, 256], key: MLXRandom.key(53)).asType(
                        dtype)
                    let mask = tril(ones([length, length])).asType(.bool)
                    let scores =
                        matmul(q.asType(.float32), k.asType(.float32).swappedAxes(-1, -2)) / 16
                    let probabilities = softmax(which(mask, scores, -Float.infinity), axis: -1)
                    let reference = matmul(probabilities, v.asType(.float32))
                    eval(q, k, v, reference)
                    return (q, k, v, reference)
                }
                let result = Device.withDefaultDevice(.gpu) {
                    MLXFast.scaledDotProductAttention(
                        queries: q, keys: k, values: v,
                        scale: 1 / 16, mask: .causal, forceFused: true)
                }
                assertRelativeError(
                    result, reference, tolerance: 0.02,
                    context: "D256 length=\(length) dtype=\(dtype)")
            }
        }
    }
}

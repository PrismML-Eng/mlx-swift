import Foundation
import MLX
import MLXNN
import XCTest

class HadamardLayerTests: XCTestCase {
    override class func setUp() { setDefaultDevice() }

    func testEmbeddingLookupAndTiedProjection() throws {
        let width = 512
        let transform = try SignedBlockHadamard(
            blockSize: 128, signs: (0 ..< width).map { $0 % 3 == 0 ? -1 : 1 })
        let weights = MLXArray(
            (0 ..< 4 * width).map { Float(($0 * 13) % 31 - 15) / 16 }, [4, width])
        for bits in [1, 2] {
            for dtype in [DType.float32, .float16, .bfloat16] {
                let (packed, scales, biases) = quantized(
                    weights.asType(dtype), groupSize: 128, bits: bits)
                let embedding = try HadamardQuantizedEmbedding(
                    weight: packed, scales: scales, biases: biases,
                    groupSize: 128, bits: bits, transform: transform)
                let unfolded = transform.inverse(
                    dequantized(
                        packed, scales: scales, biases: biases, groupSize: 128, bits: bits))
                let ids = MLXArray([3, 0, 3, 1], [2, 2])
                let lookup = embedding(ids)
                XCTAssertEqual(lookup.shape, [2, 2, width])
                XCTAssertEqual(lookup.dtype, dtype)
                XCTAssertLessThan(abs(lookup - unfolded[ids]).max().item(Float.self), 1e-5)
                let input = weights[0 ..< 2].asType(dtype)
                let reference = input.asType(.float32).matmul(unfolded.asType(.float32).T)
                let actual = embedding.asLinear(input).asType(.float32)
                let difference = actual - reference
                let relative =
                    (difference * difference).sum().sqrt()
                    / (reference * reference).sum().sqrt()
                XCTAssertLessThan(relative.item(Float.self), dtype == .float32 ? 1e-4 : 0.02)
                XCTAssertTrue(embedding.trainableParameters().flattened().isEmpty)
                XCTAssertNil(quantizeSingle(layer: embedding))
            }
        }
    }

    func testGroupedGDNBeforeTransform() throws {
        let width = 512
        let layout = try HadamardGDNLayout(width: width, keyHeads: 2, valueHeads: 4)
        let values = (0 ..< 2 * width).map { Float($0 % 71) / 71 }
        let input = MLXArray(values, [1, 2, width])
        let permutation = [0, 2, 1, 3]
        var expected = [Float]()
        for row in 0 ..< 2 {
            for head in permutation {
                expected.append(
                    contentsOf: values[
                        (row * width + head * 128) ..< (row * width + (head + 1) * 128)])
            }
        }
        let grouped = MLXArray(expected, input.shape)
        XCTAssertEqual(layout(input).asArray(Float.self), expected)
        let transform = try SignedBlockHadamard(
            blockSize: 512,
            signs: (0 ..< width).map { $0 % 5 == 0 ? -1 : 1 })
        let weights = MLXArray(values, [2, width])
        let (packed, scales, biases) = quantized(weights, groupSize: 128, bits: 2)
        let layer = try HadamardQuantizedLinear(
            weight: packed,
            bias: MLXArray([Float(0.25), -0.25]), scales: scales, biases: biases,
            groupSize: 128, bits: 2, transform: transform, gdnLayout: layout)
        let reference =
            transform(grouped).matmul(
                dequantized(
                    packed, scales: scales, biases: biases, groupSize: 128, bits: 2
                ).T)
            + MLXArray([Float(0.25), -0.25])
        XCTAssertLessThan(abs(layer(input) - reference).max().item(Float.self), 1e-4)
        XCTAssertThrowsError(try HadamardGDNLayout(width: 512, keyHeads: 3, valueHeads: 4))
    }

    func testConfigurationValidation() throws {
        var metadata: [String: Any] = [
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": 512,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": ["output.weight"],
            "prism.hadamard.inverse_weight_names": ["token_embd.weight"],
            "prism.hadamard.sign_widths": [512],
            "prism.hadamard.sign_values": Array(repeating: 1, count: 512),
            "prism.hadamard.gdn_v_grouped": true,
        ]
        func decode(_ object: [String: Any]) throws -> PrismHadamardConfiguration {
            try JSONDecoder().decode(
                PrismHadamardConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
        }
        let config = try decode(metadata)
        XCTAssertTrue(config.gdnVGrouped)
        XCTAssertEqual(try config.transform(forWidth: 512).width, 512)
        XCTAssertThrowsError(try config.transform(forWidth: 1024))
        for (key, invalid) in [
            ("prism.hadamard.version", 2),
            ("prism.hadamard.block_size", 513),
            ("prism.hadamard.sign_values", [0]),
            ("prism.hadamard.sign_widths", [Int.max]),
            ("prism.hadamard.sign_mode", "identity"),
            ("prism.hadamard.inverse_weight_names", ["output.weight"]),
        ] as [(String, Any)] {
            var candidate = metadata
            candidate[key] = invalid
            XCTAssertThrowsError(try decode(candidate), key)
        }
        metadata.removeValue(forKey: "prism.hadamard.gdn_v_grouped")
        XCTAssertFalse(try decode(metadata).gdnVGrouped)
    }

    func testRejectInvalidTransformAndPackedShape() throws {
        XCTAssertThrowsError(try SignedBlockHadamard(blockSize: 0, signs: []))
        XCTAssertThrowsError(try SignedBlockHadamard(blockSize: 16384, signs: [1]))
        XCTAssertThrowsError(try SignedBlockHadamard(blockSize: 2, signs: [1, .nan]))
        let transform = try SignedBlockHadamard(
            blockSize: 128, signs: Array(repeating: 1, count: 128))
        XCTAssertThrowsError(
            try HadamardQuantizedLinear(
                weight: MLXArray.zeros([2, 8]), scales: MLXArray.ones([2, 1]), biases: nil,
                groupSize: 128, bits: 2, transform: transform))
    }
}

// Regression guard for the M5-class (gen-17) NAX steel-gemm/qmm miscompute.
//
// The prism fork gates NAX to GPU gen-18+ (A19+) because the gen-17 NAX matmul
// path returns wrong results for fp16 GEMM at M>=8, N>=8192 (and quantized matmul
// at M>=64, N>=9216). If that gate regresses and NAX is (wrongly) enabled on a
// gen-17 device, a fp16 matmul crossing the NAX dispatch threshold diverges hard
// from the fp32 reference. This test fails in exactly that case.
//
// It is a no-op elsewhere: on GPUs where NAX is off or correct, the fp16 matmul
// tracks fp32 within normal half-precision accumulation error.

import Foundation
import MLX
import MLXRandom
import XCTest

class NAXGateRegressionTests: XCTestCase {

    override func setUp() {
        setDefaultDevice()
    }

    /// fp16 matmul over a NAX-threshold-crossing shape must match the fp32 result
    /// within half-precision accumulation error. Broken gen-17 NAX yields
    /// O(1e-1)+ relative error (vs ~1e-2 correct), so a 0.05 relative-Frobenius
    /// bound cleanly separates correct from mis-enabled NAX.
    func testNAXThresholdMatmulMatchesReference() {
        MLXRandom.seed(0)
        let m = 16, k = 512, n = 16384  // M>=8, N>=8192 -> NAX-eligible steel-gemm shape

        let a = MLXRandom.normal([m, k])
        let b = MLXRandom.normal([k, n])

        let ref = a.matmul(b)  // fp32 reference
        let got = a.asType(.float16).matmul(b.asType(.float16)).asType(.float32)

        let diffFro = ((got - ref) * (got - ref)).sum().sqrt().item(Float.self)
        let refFro = (ref * ref).sum().sqrt().item(Float.self)
        let relErr = diffFro / refFro

        XCTAssertLessThan(
            relErr, 0.05,
            "fp16 NAX-threshold matmul rel-Frobenius error \(relErr) exceeds 0.05 "
                + "(~1e-2 expected for correct fp16). NAX is likely mis-enabled on this GPU "
                + "generation — check the gen-18 gate in is_nax_available().")
    }
}

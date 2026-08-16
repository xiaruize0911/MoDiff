#pragma once
#include <cstdlib>

// =========================================================================
// THE GroupNorm group-stats block-size policy, in ONE place.
//
// WHY A HEADER AND NOT TWO COPIES. block_size sets the shape of the fp32 reduction tree, so the
// mean/inv_std it produces differ in the last bits between two different values. Two call sites have
// to agree BIT-FOR-BIT:
//
//   csrc/baseline/norm/group_norm_silu.cu   group_norm_silu_nhwc      (the fp16 GN, and the
//                                                                      "two-kernel reference")
//   csrc/modiff/norm/group_norm_silu.cu     gn_launch_group_stats     (MoDiff's fused GN->delta path)
//
// The second is verified against the first by integration/tests/gn_modiff_verify_realinput.py at ZERO
// tolerance. That file's own comment made the requirement explicit -- "block_size formula MUST match
// group_norm_silu_nhwc ... so the fp32 reduction tree -- and therefore the mean/inv_std -- is
// bit-identical to the two-kernel reference" -- and then left the formula duplicated in both files,
// which is precisely how two things that must agree stop agreeing. A prior attempt to change one side
// (a CPG-even vec2 dispatch) was reverted for failing that gate. Sharing the policy makes divergence a
// compile-time impossibility rather than a review obligation.
//
// GENERIC (what both sites did until 2026-08-16): 32 threads, doubling until it covers group_size,
//   capped at 1024.
// FAST: 128-512 threads at ~12 elements each. Measured 1.12-5.65x faster than generic on this model's
//   real GN shapes, weighted 1.91x (int8) / 2.03x (int4) -- docs/gn_fast_reduce_2026-08-16. The gain is
//   LARGEST where the shape is smallest (4.5-4.9x at 8x8 and 4x4, 1.1x at 2x2), because 1024 threads
//   are catastrophic for occupancy exactly when a group cannot fill them. The sibling
//   group_norm_silu_quantize_nhwc_impl has carried this policy behind `fast_reduce` since it was
//   written; only these two launchers were left on the generic one.
//
// NOT bit-identical to the previous build, and not claimed to be: a different reduction order moves
// mean/inv_std in the last bits, which can move a value sitting exactly on a quantize code boundary.
// What the gate requires is that the two sites agree with EACH OTHER, and they do, because they read
// this function.
//
// DEFAULT OFF (2026-08-16), and the reason is that the gate which would validate it cannot currently
// validate anything. docs/benchmark_5mode_2026-07-20/scripts/gn_modiff_verify_realinput.py is the
// ZERO-TOLERANCE differential the fused path is verified against. Five runs per arm:
//
//     MODIFF_GN_STATS_FAST=0 (generic, as shipped)   max_code_diff  35, 38, 34, 27, 36   mean 34.0
//     MODIFF_GN_STATS_FAST=1 (this fast policy)      max_code_diff  81, 42, 30, 23, 35   mean 42.2
//
// TWO CONCLUSIONS, and the second corrects a first reading of this data taken from n=1 per arm:
//
//   1. The gate FAILS on the unchanged tree, at 27-38 against a threshold of zero. The reduction change
//      that was reverted for failing it was reverted at max_code_diff=1. So this path has no working
//      correctness guarantee right now -- and nobody knew, because the gate had been UNRUNNABLE: its
//      wrapper missed the x2= the cat2 fold added on 2026-08-13, and it passed 11 arguments to a kernel
//      that had grown to 18. Both repaired.
//   2. THIS POLICY IS INDISTINGUISHABLE FROM THE GENERIC ONE ON THAT INSTRUMENT. The first measurement
//      read 35 vs 81 and looked decisive; at five runs per arm the ranges overlap (27-38 vs 23-81, the 81
//      an outlier) because the gate is NON-DETERMINISTIC across processes -- it takes a max over the
//      first 40 fused calls of a LIVE sample, and fp16 sampling here varies ~4-6e-3 between processes, so
//      the 40 calls see different data every run. A max statistic over varying inputs is exactly what
//      jumps around.
//
// So the gate is evidence AGAINST NEITHER policy, and evidence that it cannot gate. Default stays OFF as
// a judgement rather than a measurement: this path's only correctness guard is itself broken, and
// changing the path while that is true is not justified. The prize -- a predicted +7.80/+8.16 ms/step on
// the MoDiff arms, by analogy with the PTQ family's verified 1.91x/2.03x -- stays UNBANKED until
// docs/OPEN_ITEMS.md A0 closes. Set MODIFF_GN_STATS_FAST=1 to measure it.
//
// Read ONCE per process (static local in an inline function -- one instance across translation units):
// if the two sites could ever observe different values of the flag they would silently diverge, and the
// gate would only catch it if somebody happened to run it.
// =========================================================================
inline int modiff_gn_stats_block_size(long group_size) {
    static const bool fast = []() {
        const char* e = std::getenv("MODIFF_GN_STATS_FAST");
        return e != nullptr && e[0] == '1';
    }();
    int block_size;
    if (fast) {
        block_size = 128;
        while ((long)block_size * 12 < group_size && block_size < 512) block_size <<= 1;
    } else {
        block_size = 32;
        while (block_size < group_size && block_size < 1024) block_size <<= 1;
    }
    return block_size;
}

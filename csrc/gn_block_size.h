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
// DEFAULT OFF (2026-08-16) -- AND THE PREMISE IS REFUTED, so it should stay off.
//
// This policy only affects gn_launch_group_stats' GROUP-MAJOR fallback. The MoDiff path does not take it:
// gn_launch_group_stats defaults to a CHANMAJOR decomposition (BLK = C/K) and returns before reaching the
// group-major code. So on the MoDiff arms this flag is inert -- confirmed by a cross-arm output comparison
// (integration/tests/gn_cross_arm_check.py): byte-identical, 0/40 cases differ.
//
// Forcing the group-major path (MODIFF_GN_STATS_ALT=0) to measure what the policy WOULD buy, summed over
// this model's real GN shapes:
//
//                          batch 8      batch 128 (production)
//   chanmajor (default)     223.6 us      1050.4 us   <- fastest where it matters
//   group-major + fast      179.7 us      1060.0 us
//   group-major + generic   204.2 us      1662.2 us
//
// THE RANKING INVERTS WITH BATCH, and batch is a first-order variable here: chanmajor's block size is C/K,
// independent of batch, while group-major's GRID is N*num_groups -- 256 blocks at batch 8 on 84 SMs
// (occupancy-starved) against 4096 at batch 128. A 1.24x win at batch 8 becomes a 1% loss at 128, and
// end-to-end agrees: the MoDiff arm measured 73.9 vs 73.2 ms/step, i.e. 0.7 ms/step SLOWER, 3 runs of 3.
//
// So chanmajor is already the right decomposition at production batch, and C1's 1.91x does NOT generalise
// here: C1 sped up the PTQ family, which IS group-major, where fast_reduce genuinely wins at batch 128.
// The analogy that produced C10's predicted +7.80/+8.16 ms/step compared the wrong two things.
//
// Kept because it makes the shared-policy invariant structural (two launchers that must agree bit-for-bit
// now read one function instead of duplicating a formula), and because the group-major fallback still
// exists and is still selectable. All three decompositions score max|d| = 1.0 against an fp64
// reconstruction, i.e. fp16 rounding -- see docs/OPEN_ITEMS.md A0.
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

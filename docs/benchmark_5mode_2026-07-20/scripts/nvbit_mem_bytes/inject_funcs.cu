#include <stdint.h>
#include <stdio.h>
#include "utils/utils.h"

/* Per warp-level global memory instruction: add (active_threads * access_size) bytes to the
 * read or write counter. One atomicAdd per warp (first active lane) -> low overhead. */
extern "C" __device__ __noinline__ void count_bytes(int pred, int is_write, int size,
                                                    uint64_t p_rd, uint64_t p_wr) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), pred);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(predicate_mask);
    if (first_laneid == laneid && num_threads > 0) {
        unsigned long long b = (unsigned long long)num_threads * (unsigned long long)size;
        atomicAdd((unsigned long long*)(is_write ? p_wr : p_rd), b);
    }
}

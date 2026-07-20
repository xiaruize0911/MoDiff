/* NVBit tool: per-kernel measured GLOBAL-memory read/write bytes (no HW perf counters).
 * Instruments each global load/store with count_bytes (active_threads * access_size). Prints
 * one line per kernel launched inside a cuProfilerStart/Stop region (ACTIVE_FROM_START=0):
 *   MEMBYTES read=<bytes> write=<bytes> blocks=<n> kernel=<name>
 * Read/write split from the opcode: LD prefix = read, ST/RED/ATOM = write. */
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <sstream>
#include <algorithm>

/* every tool needs to include this once */
#include "nvbit_tool.h"
/* nvbit interface file */
#include "nvbit.h"
/* nvbit utility functions */
#include "utils/utils.h"

__managed__ uint64_t g_bytes_read = 0;
__managed__ uint64_t g_bytes_write = 0;
uint64_t tot_read = 0, tot_write = 0;
uint32_t kernel_id = 0;
int verbose = 0, active_from_start = 1, mangled = 1;
bool active_region = true;
pthread_mutex_t mutex;
std::unordered_set<CUfunction> already_instrumented;

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(active_from_start, "ACTIVE_FROM_START", 1,
                "Count from start (1) or only between cuProfilerStart/Stop (0)");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1, "Print kernel names mangled or not");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    if (active_from_start == 0) active_region = false;
    std::string pad(80, '-'); printf("%s\n", pad.c_str());
}

static inline bool is_write_op(const char* op) {
    return (op[0] == 'S' && op[1] == 'T') || strncmp(op, "RED", 3) == 0 || strncmp(op, "ATOM", 4) == 0;
}
static inline bool is_read_op(const char* op) {
    return (op[0] == 'L' && op[1] == 'D') || strncmp(op, "ATOM", 4) == 0;
}

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> rf = nvbit_get_related_functions(ctx, func);
    rf.push_back(func);
    for (auto f : rf) {
        if (!already_instrumented.insert(f).second) continue;
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto i : instrs) {
            if (i->getMemorySpace() != InstrType::MemorySpace::GLOBAL) continue;   // DRAM global only
            const char* op = i->getOpcodeShort();
            int is_w = is_write_op(op) ? 1 : 0;
            if (!is_w && !is_read_op(op)) continue;
            int size = i->getSize();                                               // bytes/thread
            nvbit_insert_call(i, "count_bytes", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(i);
            nvbit_add_call_arg_const_val32(i, is_w);
            nvbit_add_call_arg_const_val32(i, size);
            nvbit_add_call_arg_const_val64(i, (uint64_t)&g_bytes_read);
            nvbit_add_call_arg_const_val64(i, (uint64_t)&g_bytes_write);
        }
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx)
            func = ((cuLaunchKernelEx_params*)params)->f;
        else
            func = ((cuLaunchKernel_params*)params)->f;

        if (!is_exit) {
            pthread_mutex_lock(&mutex);
            instrument_function_if_needed(ctx, func);
            nvbit_enable_instrumented(ctx, func, active_region);
            g_bytes_read = 0; g_bytes_write = 0;
        } else {
            CUDA_SAFECALL(cudaDeviceSynchronize());
            if (active_region) {
                tot_read += g_bytes_read; tot_write += g_bytes_write;
                int ctas = 0;
                if (cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernel_ptsz) {
                    cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
                    ctas = p->gridDimX * p->gridDimY * p->gridDimZ;
                } else if (cbid == API_CUDA_cuLaunchKernelEx || cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
                    cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
                    ctas = p->config->gridDimX * p->config->gridDimY * p->config->gridDimZ;
                }
                printf("MEMBYTES read=%lu write=%lu blocks=%d kernel=%s\n",
                       (unsigned long)g_bytes_read, (unsigned long)g_bytes_write, ctas,
                       nvbit_get_func_name(ctx, func, mangled));
                kernel_id++;
            }
            pthread_mutex_unlock(&mutex);
        }
    } else if (cbid == API_CUDA_cuProfilerStart && is_exit) {
        if (!active_from_start) active_region = true;
    } else if (cbid == API_CUDA_cuProfilerStop && !is_exit) {
        if (!active_from_start) active_region = false;
    }
}

void nvbit_at_term() {
    printf("MEMBYTES_TOTAL read=%lu write=%lu\n", (unsigned long)tot_read, (unsigned long)tot_write);
}

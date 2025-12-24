#ifndef PROFILER_H
#define PROFILER_H

#include <stdint.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

#define NUM_OF_KERNELS 5 

typedef enum {
    DOT_PRODUCT_VEC4,
    REDUCE_SUM4_FLOAT4_SLIDING,
    UPDATE_R_AND_Z,
    UPDATE_X_AND_P,
    OBM_MATVEC_MULT
} Kernels;

extern const char *const kNames[NUM_OF_KERNELS];

typedef struct{
    uint64_t count;
    double total_time;
    double time_mean;
    double time_m2;
    double min_time;
    double max_time;
} KernelStats;

typedef struct{
    uint32_t iterations;
    uint32_t curr_iteration;
    KernelStats (*kernels)[NUM_OF_KERNELS];

    double cn_elapsed;

    uint32_t *cg_iterations;
    double *cg_elapsed;
} Profiler;

void profiler_init(Profiler *p, uint32_t iterations);
void kernelstats_init(Profiler *p, uint32_t iteration);
KernelStats kernelstats_get();
void profile_kernel(Profiler *p, Kernels kernel, cl_event event, uint32_t iteration);
double get_kernel_time(cl_event event);
void add_kernel_sample(Profiler *p, Kernels kernel, double time_sample, uint32_t iteration);
void save_profiler_json(Profiler *p, const char *filename);
#endif

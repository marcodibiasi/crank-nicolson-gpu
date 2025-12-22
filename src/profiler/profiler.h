#ifndef PROFILER_H
#define PROFILER_H

#include <stdint.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

#define NUM_OF_KERNELS 6

typedef enum {
    DOT_PRODUCT_VEC4,
    REDUCE_SUM4_FLOAT4_SLIDING,
    UPDATE_R_AND_Z,
    UPDATE_X_AND_P,
    OBM_MATVEC_MULT,
    UPDATE_B
} Kernels;

typedef struct{
    uint64_t count;
    double total_time;
    double time_mean;
    double time_m2;
    double min_time;
    double max_time;
} KernelStats;

typedef struct{
    KernelStats kernels[NUM_OF_KERNELS];

    uint32_t iterations;
    double *cn_elapsed;

    uint32_t *cg_iterations;
    double *cg_elapsed;

    double *saving_time;
} Profiler;

void profiler_init(Profiler *p, uint32_t iterations);
KernelStats kernelstats_init();
void profile_kernel(Profiler *p, Kernels kernel, cl_event event);
double get_kernel_time(cl_event event);
void add_kernel_sample(Profiler *p, Kernels kernel, double time_sample);
#endif

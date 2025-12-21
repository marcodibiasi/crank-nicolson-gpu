#ifndef PROFILER_H
#define PROFILER_H

#include <stdint.h>

#define NUM_OF_KERNELS 5

typedef enum {
    DOT_PRODUCT_VEC4,
    REDUCE_SUM4_FLOAT4_SLIDING,
    UPDATE_R_AND_Z,
    UPDATE_X_AND_P,
    OBM_MATVEC_MULT
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
#endif

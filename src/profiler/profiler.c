#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "profiler.h"

void profiler_init(Profiler *p, uint32_t iterations){
    p->iterations = iterations;
    p->cn_elapsed = malloc(sizeof(double) * iterations);
    p->cg_iterations = malloc(sizeof(uint32_t) * iterations);
    p->cg_elapsed = malloc(sizeof(double) * iterations);
    p->saving_time = malloc(sizeof(double) * iterations);

    p->kernels[DOT_PRODUCT_VEC4] = kernelstats_init();
    p->kernels[REDUCE_SUM4_FLOAT4_SLIDING] = kernelstats_init(); 
    p->kernels[UPDATE_R_AND_Z] = kernelstats_init();
    p->kernels[UPDATE_X_AND_P] = kernelstats_init();
    p->kernels[OBM_MATVEC_MULT] = kernelstats_init();
}

KernelStats kernelstats_init(){
    KernelStats ks = {
        .count = 0,
        .total_time = 0.0,
        .time_mean = 0.0,
        .time_m2 = 0.0,
        .min_time = 0.0,
        .max_time = 0.0
    };

    return ks;
}

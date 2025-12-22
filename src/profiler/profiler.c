#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "profiler.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

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
    p->kernels[UPDATE_B] = kernelstats_init();
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

void profile_kernel(Profiler *p, Kernels kernel, cl_event event){
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

    double current_exec_time = (double) (end - start) / 1e6;

    printf("DEBUG: %lf ms\n", current_exec_time);
}

double get_kernel_time(cl_event event){
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

    return (double) (end - start) / 1e6;
}

void add_kernel_sample(Profiler *p, Kernels kernel, double time_sample){ 
    printf("DEBUG: %lf ms\n", time_sample);
}

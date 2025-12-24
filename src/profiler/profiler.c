#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include "cJSON.h"
#include "profiler.h"


#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

const char *const kNames[] = {
    [DOT_PRODUCT_VEC4] = "DOT_PRODUCT_VEC4",
    [REDUCE_SUM4_FLOAT4_SLIDING] = "REDUCE_SUM4_FLOAT4_SLIDING",
    [UPDATE_R_AND_Z] = "UPDATE_R_AND_Z",
    [UPDATE_X_AND_P] = "UPDATE_X_AND_P",
    [OBM_MATVEC_MULT] = "OBM_MATVEC_MULT"
};

void profiler_init(Profiler *p, uint32_t iterations){
    p->iterations = iterations;
    p->curr_iteration = 0;
    p->kernels = malloc(sizeof(*p->kernels) * iterations);
    p->cn_elapsed = 0;
    p->cg_iterations = malloc(sizeof(uint32_t) * iterations);
    p->cg_elapsed = malloc(sizeof(double) * iterations);
    
    kernelstats_init(p, 0);
}

void kernelstats_init(Profiler *p, uint32_t iteration){
    p->kernels[iteration][DOT_PRODUCT_VEC4] = kernelstats_get();
    p->kernels[iteration][REDUCE_SUM4_FLOAT4_SLIDING] = kernelstats_get(); 
    p->kernels[iteration][UPDATE_R_AND_Z] = kernelstats_get();
    p->kernels[iteration][UPDATE_X_AND_P] = kernelstats_get();
    p->kernels[iteration][OBM_MATVEC_MULT] = kernelstats_get();
}

KernelStats kernelstats_get(){
    KernelStats ks = {
        .count = 0,
        .total_time = 0.0,
        .time_mean = 0.0,
        .time_m2 = 0.0,
        .min_time = DBL_MAX,
        .max_time = 0.0
    };

    return ks;
}

void profile_kernel(Profiler *p, Kernels kernel, cl_event event, uint32_t iteration){
    double current_exec_time = get_kernel_time(event);
    add_kernel_sample(p, kernel, current_exec_time, iteration);
}

double get_kernel_time(cl_event event){
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

    return (double) (end - start) / 1e6;
}

void add_kernel_sample(Profiler *p, Kernels kernel, double time_sample, uint32_t iteration){ 
    // Handle count, total_time, min_time and max_time
    p->kernels[iteration][kernel].count++;
    p->kernels[iteration][kernel].total_time += time_sample;
    if(p->kernels[iteration][kernel].min_time > time_sample) 
        p->kernels[iteration][kernel].min_time = time_sample;
    if(p->kernels[iteration][kernel].max_time < time_sample) 
        p->kernels[iteration][kernel].max_time = time_sample;

    // Handle time_mean and time_m2
    double delta = time_sample - p->kernels[iteration][kernel].time_mean;
    p->kernels[iteration][kernel].time_mean += delta / p->kernels[iteration][kernel].count; 
    double delta2 = time_sample - p->kernels[iteration][kernel].time_mean;
    p->kernels[iteration][kernel].time_m2 += delta * delta2;
}


void save_profiler_json(Profiler *p, const char *filename) {
    cJSON *root = cJSON_CreateObject();

    cJSON_AddStringToObject(root, "kernel_unit", "ms");
    cJSON_AddStringToObject(root, "total_time_unit", "s");
    
    cJSON_AddNumberToObject(root, "iterations", p->iterations);
    cJSON_AddNumberToObject(root, "cn_elapsed", p->cn_elapsed);

    // cg_iterations
    cJSON *cg_iter_array = cJSON_CreateIntArray((const int*)p->cg_iterations, p->iterations);
    cJSON_AddItemToObject(root, "cg_iterations", cg_iter_array);

    // cg_elapsed
    cJSON *cg_elapsed_array = cJSON_CreateDoubleArray(p->cg_elapsed, p->iterations);
    cJSON_AddItemToObject(root, "cg_elapsed", cg_elapsed_array);

    // kernels
    cJSON *kernels_array = cJSON_CreateArray();
    for (uint32_t i = 0; i < p->iterations; i++) {
        cJSON *iteration_obj = cJSON_CreateObject();

        for (int k = 0; k < NUM_OF_KERNELS; k++) {
            cJSON *ks_obj = cJSON_CreateObject();
            KernelStats *ks = &p->kernels[i][k];

            cJSON_AddNumberToObject(ks_obj, "count", ks->count);
            cJSON_AddNumberToObject(ks_obj, "total_time", ks->total_time);
            cJSON_AddNumberToObject(ks_obj, "time_mean", ks->time_mean);
            cJSON_AddNumberToObject(ks_obj, "time_m2", ks->time_m2);
            cJSON_AddNumberToObject(ks_obj, "min_time", ks->min_time);
            cJSON_AddNumberToObject(ks_obj, "max_time", ks->max_time);

            cJSON_AddItemToObject(iteration_obj, kNames[k], ks_obj); // kNames = nomi dei kernel
        }

        cJSON_AddItemToArray(kernels_array, iteration_obj);
    }
    cJSON_AddItemToObject(root, "kernels", kernels_array);

    char *json_str = cJSON_Print(root);

    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "%s", json_str);
        fclose(f);
    } else {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
    }

    cJSON_free(json_str);
    cJSON_Delete(root);
}

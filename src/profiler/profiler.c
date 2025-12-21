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
}

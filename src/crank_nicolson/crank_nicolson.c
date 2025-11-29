#include "crank_nicolson.h"
#include "conjugate_gradient.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "flags.h"

#define RESET   "\033[0m"
#define TITLE   "\033[92m" 
#define LABEL   "\033[32m"  

CrankNicolsonSetup *setup(int size, float dx, float dt, float alpha, float *u_curr){
    CrankNicolsonSetup *cn_solver = malloc(sizeof(CrankNicolsonSetup));
    if(!cn_solver) return NULL;

    cn_solver->size = size;
    cn_solver->dx = dx;
    cn_solver->dt = dt;
    cn_solver->alpha = alpha;
    cn_solver->u_current = u_curr;

    //This version only supports square heatmaps
    cn_solver->rx = (alpha * dt) / (dx * dx);
    if (2 * cn_solver->rx > 1.0) {
        fprintf(stderr, 
            "Warning: 2 * rx = %.2f > 1.0 — numerical oscillations may arise.\n",
            2 * cn_solver->rx);
    }

    cn_solver->time_step = 0;

    cn_solver->A = define_matrix(cn_solver->rx, cn_solver->size);
    cn_solver->B = define_matrix(-cn_solver->rx, cn_solver->size);
    cn_solver->cg_solver = setup_solver(size, cn_solver->A, u_curr, NULL);
    cn_solver->b = calculate_unknown_vector(cn_solver->cg_solver.cl, cn_solver->B, u_curr);
    update_unknown_b(&cn_solver->cg_solver, cn_solver->b);

    printf(TITLE "\nSolver settings:\n" RESET
       LABEL "\tSize: " RESET "%d\n"
       LABEL "\tdx: " RESET "%.5f\n"
       LABEL "\trx: " RESET "%.5f\n"
       LABEL "\tdt: " RESET "%.5f\n"
       LABEL "\talpha: " RESET "%.5f\n\n",
       size, dx, cn_solver->rx, dt, alpha);

    return cn_solver;
}

OBMatrix define_matrix(float clr, int size) {
    int Nx = (int)sqrt((double)size);
    int *offsets = malloc(5 * sizeof(int));
    float *values = malloc(5 * sizeof(float));

    offsets[0] = -Nx; 
    offsets[1] = -1; 
    offsets[2] = 0; 
    offsets[3] = 1; 
    offsets[4] = Nx;

    values[0] = values[1] = values[3] = values[4] = -clr/2;
    values[2] = 1 + 2*clr;

    return (OBMatrix){
        .rows = size,
        .offset = offsets,
        .values = values,
        .non_zero_values = 5
    };
}

void run(CrankNicolsonSetup *solver, int iterations, Flags *flags){
    float elapsed = 0.0f;

    for (int i = 0; i < iterations; i++){
        elapsed += iterate(solver, flags);
        float* new_b = calculate_unknown_vector(solver->cg_solver.cl, solver->B, solver->cg_solver.x);
        memcpy(solver->b, new_b, solver->size * sizeof(float));
        free(new_b);
        update_unknown_b(&solver->cg_solver, solver->b);
    }

    printf(TITLE "\nSimulation completed.\n" RESET
       LABEL "Total elapsed time: " RESET "%.3f s\n"
       LABEL "Average time per iteration: " RESET "%.3f s\n\n",
       elapsed, elapsed / iterations);
}

float iterate(CrankNicolsonSetup *solver, Flags *flags) {
    float elapsed = conjugate_gradient(&solver->cg_solver, flags);
    
    // Saving
    char path_png[256];
    sprintf(path_png, "data/img/t%d.png", solver->time_step);
    unsigned char* vec = pgm_denormalisation(solver->cg_solver.x, solver->size);
    png_save(path_png, vec, solver->size, flags->verbose);
    
    // char path_pgm[256];
    // sprintf(path_pgm, "data/simulation/t%d.pgm", solver->time_step);
    // pgm_save(path_pgm, vec, solver->size);

    solver->time_step++;

    return elapsed;
}

void free_solver(CrankNicolsonSetup *solver) {
    if(solver->u_current) free(solver->u_current);
    if(solver->b) free(solver->b);
    if(solver->A.offset) free(solver->A.offset);
    if(solver->A.values) free(solver->A.values);
    if(solver->B.offset) free(solver->B.offset);
    if(solver->B.values) free(solver->B.values);

    free_cg_solver(&solver->cg_solver);
    free(solver);
}

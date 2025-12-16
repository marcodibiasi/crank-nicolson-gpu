#include "crank_nicolson.h"
#include "conjugate_gradient.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
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
            "Warning: 2 * rx = %.2f > 1.0 - numerical oscillations may arise.\n",
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
    BufferPool buffs = {
        .n = 2, 
        .buf_size = solver->size
    };
    buffs.data = malloc(sizeof(float) * buffs.n * buffs.buf_size);

    ThreadArgs cn_shared_data = {
        .solver = solver,
        .iterations = iterations,
        .flags = flags,
        .buffs = &buffs,
        .buff_ready = 0,
        .which_buf = 0
    };
    pthread_mutex_init(&cn_shared_data.mutex, NULL);
    pthread_cond_init(&cn_shared_data.can_read, NULL);

    pthread_t conjugate_gradient_t, save_frame_t;
    if(pthread_create(&conjugate_gradient_t, NULL, run_conjugate_gradient, (void*)&cn_shared_data) == 1){
        perror("pthread_create");
        exit(EXIT_FAILURE);
    }

    if(pthread_create(&save_frame_t, NULL, save_frame, (void*)&cn_shared_data) == 1){
        perror("pthread_create");
        exit(EXIT_FAILURE);
    }

    pthread_join(conjugate_gradient_t, NULL);
    pthread_join(save_frame_t, NULL);
}


void* run_conjugate_gradient(void* arg){
    ThreadArgs* t_args = (ThreadArgs*)arg;
    CrankNicolsonSetup* solver = t_args->solver;
    BufferPool* buffs = t_args->buffs;
    float cg_elapsed = 0.0f;
    float* ptr_to_save;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start); 

    while(solver->time_step < t_args->iterations){
        cg_elapsed += conjugate_gradient(&solver->cg_solver, t_args->flags);
      
        // Critical section: saving device buffer into host array
        pthread_mutex_lock(&t_args->mutex);
        
        ptr_to_save = buffs->data + (t_args->which_buf * buffs->buf_size);  
        save_result(&solver->cg_solver, solver->size, ptr_to_save); 
        t_args->buff_ready = 1;

        pthread_cond_signal(&t_args->can_read);
        pthread_mutex_unlock(&t_args->mutex);
        
        // Update unknown b
        float* new_b = calculate_unknown_vector(solver->cg_solver.cl, solver->B, ptr_to_save);
        memcpy(solver->b, new_b, solver->size * sizeof(float));
        free(new_b);
        update_unknown_b(&solver->cg_solver, solver->b);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    float cn_elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf(TITLE "\nSimulation completed.\n" RESET
        LABEL "Total elapsed time: " RESET "%.3f s\n"
        LABEL "Total PCG time: " RESET "%.3f s\n"
        LABEL "Average time per PCG iteration: " RESET "%.3f s\n\n",
        cn_elapsed, cg_elapsed, cg_elapsed / t_args->iterations);

    return NULL;
}

void* save_frame(void* args){
    ThreadArgs *t_args = (ThreadArgs*)args;
    CrankNicolsonSetup* solver = t_args->solver;
    BufferPool *buffs = t_args->buffs; 
    char path_png[256];
    float* ptr_to_save;
    
    float percent = 0.0f;
    while(solver->time_step < t_args->iterations){
        if(t_args->flags->progress == 1){
            percent = solver->time_step * 100 / t_args->iterations;
            fprintf(stdout, "\rPercent: %3.1f%%", percent);
            fflush(stdout);
        }

        // Critical section: waiting for host array to be ready to save the png
        pthread_mutex_lock(&t_args->mutex);
        while(!t_args->buff_ready){
            pthread_cond_wait(&t_args->can_read, &t_args->mutex);
        }
        
        // Saving
        ptr_to_save = buffs->data + (t_args->which_buf * buffs->buf_size);
        sprintf(path_png, "data/img/t%d.png", solver->time_step);
        unsigned char *vec = pgm_denormalisation(ptr_to_save, solver->size);
        png_save(path_png, vec, solver->size, t_args->flags->verbose);
     
        // Switch buffer
        t_args->which_buf = (t_args->which_buf + 1) % buffs->n;
        t_args->buff_ready = 0;

        solver->time_step++;

        pthread_mutex_unlock(&t_args->mutex);
    }

    return NULL;
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

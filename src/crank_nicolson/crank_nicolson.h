#ifndef CRANK_NICOLSON_H
#define CRANK_NICOLSON_H

#include <pthread.h>
#include <semaphore.h>
#include "obm.h"
#include "conjugate_gradient.h"
#include "pgm_utils.h"
#include "flags.h"
#include "profiler.h"

typedef struct {
    int size;   // Represents the length of the flattened image (Es. for img = 1024 x 1024, size = 1024**2)
    int time_step;  // Curent simulation index (starts from 0)
    float dx;  // Physical distance 
    float dt;      // Physical time between steps
    float rx;  // Numerical diffusion coefficients 
    float alpha;   // Diffusion coefficient

    float *u_current;  // Current state of the system

    float *b;   // Unknown vector
    OBMatrix A;  // Banded matrix in Offsetted format
    OBMatrix B;  // Matrix used to compute b (b = B * u^k)

    Solver cg_solver;  //Conjugate Gradient solver instance

} CrankNicolsonSetup;

// data has length n * buf_size
typedef struct{
    float *data;
    int buf_size;
    int n;

    int head; // producer write
    int tail; // consumer read 
} BufferPool;

typedef struct{
    CrankNicolsonSetup* solver;
    int iterations;
    Flags *flags;

    BufferPool *buffs;
    Profiler *profiler;

    sem_t empty;
    sem_t full;
    pthread_mutex_t mutex;
} ThreadArgs;

CrankNicolsonSetup *setup(int size, float dx, float dt, float alpha, float *u_curr);
OBMatrix define_matrix(float clr, int size);
void run(CrankNicolsonSetup *solver, int iterations, Flags *flags, Profiler *p);
void free_solver(CrankNicolsonSetup *solver);
void* run_conjugate_gradient(void* arg);
void* save_frame(void* arg);

#endif

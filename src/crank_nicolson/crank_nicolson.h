#ifndef CRANK_NICOLSON_H
#define CRANK_NICOLSON_H

#include <pthread.h>
#include "obm.h"
#include "conjugate_gradient.h"
#include "pgm_utils.h"
#include "flags.h"


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
    int n;
    int buf_size;
    float* data;
} BufferPool;

typedef struct{
    CrankNicolsonSetup* solver;
    int iterations;
    Flags* flags;

    BufferPool* buffs;

    // Signal if the second thread can save the image
    int buff_ready;

    // Indicates which buffer from the pool should be saved
    int which_buf;

    pthread_mutex_t mutex;
    pthread_cond_t can_read;
} ThreadArgs;

CrankNicolsonSetup *setup(int size, float dx, float dt, float alpha, float *u_curr);
OBMatrix define_matrix(float clr, int size);
void run(CrankNicolsonSetup *solver, int iterations, Flags *flags);
void free_solver(CrankNicolsonSetup *solver);
void* run_conjugate_gradient(void* arg);
void* save_frame(void* arg);

#endif

#ifndef CRANK_NICOLSON_H
#define CRANK_NICOLSON_H

#include "obm.h"
#include "conjugate_gradient.h"
#include "pgm_utils.h"


typedef struct {
    int size;   //Represents the length of the flattened image (Es. for img = 1024 x 1024, size = 1024**2)
    int time_step;  //Curent simulation index (starts from 0)
    float dx;  //Physical distance 
    float dt;      //Physical time between steps
    float rx;  //Numerical diffusion coefficients 
    float alpha;   //Diffusion coefficient

    float *u_current;  //Current state of the system

    float *b;   // Unknown vector
    OBMatrix A;  // Banded matrix in Offsetted format
    OBMatrix B;  // Matrix used to compute b (b = B * u^k)

    Solver cg_solver;  //Conjugate Gradient solver instance
} CrankNicolsonSetup;

CrankNicolsonSetup *setup(int size, float dx, float dt, float alpha, float *u_curr);
OBMatrix define_matrix(float clr, int size);
void run(CrankNicolsonSetup *solver, int iterations);
void iterate(CrankNicolsonSetup *solver);
void free_solver(CrankNicolsonSetup *solver);

#endif
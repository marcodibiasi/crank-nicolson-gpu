#ifndef CONJUGATE_GRADIENT_H    
#define CONJUGATE_GRADIENT_H

#include "obm.h"
#include "flags.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

typedef struct{
    cl_kernel reduce_sum4_float4_sliding;
    cl_kernel update_x;
    cl_kernel update_r_and_z;
    cl_kernel update_p;
    cl_kernel dot_product_vec4;
    cl_kernel obm_matvec_mult;
} OpenCLKernels;

typedef struct{
    cl_platform_id p;
	cl_device_id d;
	cl_context ctx;
	cl_command_queue q;
	cl_program prog;

    cl_mem x_buffer; 
    cl_mem b_buffer;

    cl_int obm_rows;
    cl_mem obm_offset_buffer;
    cl_mem obm_values_buffer;

    size_t lws;

    OpenCLKernels kernels;
} OpenCLContext;


typedef struct {
    int size;

    OBMatrix A_obm;

    float *x; 
    float *b;    
    
    OpenCLContext cl;
} Solver;

Solver setup_solver(int size, OBMatrix A_obm, float *b, float *initial_x);
OpenCLContext setup_opencl_context(Solver* solver);
float conjugate_gradient(Solver* solver, Flags *flags);
float alpha_calculate(Solver* solver, cl_mem* r, cl_mem* z, cl_mem* p, cl_mem* Ap, float *r_dot_z, Flags *flags);
cl_event dot_product_vec4(Solver *solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int length);
float dot_product_handler(Solver *solver, cl_mem *vec1, cl_mem *vec2, int length, Flags *flags);
cl_event reduce_sum4_float4_sliding(Solver *solver, cl_mem* vec_in, cl_mem* vec_out, int length);
cl_event obm_matvec_mult(Solver* solver, cl_mem* vec, cl_mem* result);
cl_event update_x(Solver *solver, cl_mem* p, float alpha, int length);
cl_event update_p(Solver *solver, cl_mem* p, cl_mem* z, float beta, int length);
cl_event update_r_and_z(Solver* solver, cl_mem* r, cl_mem* Ap, cl_mem* precond, cl_mem* r_next, cl_mem* z_next, float alpha, int length);
void free_cg_solver(Solver* solver);
void save_result(Solver *solver, cl_mem buf, size_t size, int n);
float profiling_event(cl_event event);

// Support functions for the Crank Nicolson solver
float* calculate_unknown_vector(OpenCLContext cl, OBMatrix B, float* u_n);
void update_unknown_b(Solver* solver, float* b);

#endif // CONJUGATE_GRADIENT_H 
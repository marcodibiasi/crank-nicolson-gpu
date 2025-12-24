#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <time.h>
#include "conjugate_gradient.h"
#include "ocl_boiler.h"
#include "flags.h"
#include "profiler.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif


Solver setup_solver(int size, OBMatrix A_obm, float *initial_b, float *initial_x) {
    Solver solver;
    solver.size = size;
    solver.A_obm = A_obm;
    solver.b = initial_b;
    solver.x = initial_x;

    if(!solver.b) {
        printf("Initializing zero-vector b\n");
        solver.b = malloc(size * sizeof(float));
        if(!solver.b){
            fprintf(stderr, "Failed to allocate memory for b vector\n");
            exit(EXIT_FAILURE);
        }
        for(int i = 0; i < size; i++){
            solver.b[i] = 0.0f;
        }
    }

    if (!solver.x) {
        printf("Initializing zero-vector x\n");
        solver.x = malloc(size * sizeof(float));
        if (!solver.x){
            fprintf(stderr, "Failed to allocate memory for x vector\n");
            exit(1);
        }
        for (int i = 0; i < size; i++){
            solver.x[i] = 0.0f;
        }
    }

    solver.cl = setup_opencl_context(&solver);  

    /*
    Precondition with Jacobi by extracting the diagonal of the matrix A
    and inverting it
    */
    //symmetrical offsets
    
    // The better alternative would be clEnqueueFillBuffer, but MacOS does not support it. 
    // To make the software crossplatfrom, clEnqueueWriteBuffer has been used
    cl_int err;
    float diagonal_value = 1.0f / solver.A_obm.values[solver.A_obm.non_zero_values/2];
    float *diag = malloc(sizeof(float) * solver.size);
    for (int i = 0; i < solver.size; ++i) diag[i] = diagonal_value;

    err = clEnqueueWriteBuffer(solver.cl.q, solver.cl.temp.diagonal_buffer, CL_TRUE, 0,
            sizeof(float) * solver.size, diag, 0, NULL, NULL);

    free(diag);
    ocl_check(err, "clEnqueueWriteBuffer failed for diagonal_buffer");

    return solver;
}

OpenCLContext setup_opencl_context(Solver* solver) {
    OpenCLContext cl;

    cl.p = select_platform();
	cl.d = select_device(cl.p);
	cl.ctx = create_context(cl.p, cl.d); 
	cl.q = create_queue(cl.ctx, cl.d);
	cl.prog = create_program("src/conjugate_gradient/kernels.cl", cl.ctx, cl.d);

	cl_int err;

    // Allocate OpenCL buffers
	cl.b_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        solver->size * sizeof(float), solver->b, &err);
	ocl_check(err, "clCreateBuffer failed for b_buffer");

    cl.x_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        solver->size * sizeof(float), solver->x, &err);
    ocl_check(err, "clCreateBuffer failed for x_buffer");

    // OBM representation
    cl.obm_rows = solver->A_obm.rows;
    cl.obm_offset_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        solver->A_obm.non_zero_values * sizeof(int), solver->A_obm.offset, &err);
    cl.obm_values_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        solver->A_obm.non_zero_values * sizeof(float), solver->A_obm.values, &err);
    
    float B_obm_values[] = {
        -solver->A_obm.values[0],
        -solver->A_obm.values[1],
        2 + (-solver->A_obm.values[2]),
        -solver->A_obm.values[3],
        -solver->A_obm.values[4]
    };

    cl.B_obm_values_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        solver->A_obm.non_zero_values * sizeof(float), B_obm_values, &err);

    cl.temp = temporary_buffers_init(&cl, solver->size);

    // Create kernels
    cl.kernels.reduce_sum4_float4_sliding = clCreateKernel(cl.prog, "reduce_sum4_float4_sliding", &err);
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.dot_product_vec4 = clCreateKernel(cl.prog, "dot_product_vec4", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.update_r_and_z = clCreateKernel(cl.prog, "update_r_and_z", &err);
    ocl_check(err, "clCreateKernel failed for update_r_and_z");
    
    cl.kernels.update_x_and_p = clCreateKernel(cl.prog, "update_x_and_p", &err);
    ocl_check(err, "clCreateKernel failed for update_x_and_p");

    cl.kernels.obm_matvec_mult = clCreateKernel(cl.prog, "obm_matvec_mult", &err);
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.obm_matvec_mult_local = clCreateKernel(cl.prog, "obm_matvec_mult_local", &err);
    ocl_check(err, "clCreateKernel failed");

    cl.lws = 128;

    return cl;
}

TemporaryBuffers temporary_buffers_init(OpenCLContext *cl, int length) {
    TemporaryBuffers temp;
    cl_int err;

    temp.diagonal_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err); 
    ocl_check(err, "clCreateBuffer failed for diagonal_buffer");
    
    temp.r_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for r_buffer");
    
    temp.Ap = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for Ap");

    temp.z_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for z_buffer");
    
    temp.p_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for p_buffer");

    temp.r_next_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for r_next_buffer");

    temp.z_next_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for z_next_buffer");

    return temp;
}

float conjugate_gradient(Solver* solver, Flags *flags, Profiler *p) {
    cl_int err;
    OpenCLContext* cl = &solver->cl;
    TemporaryBuffers *temp = &cl->temp;

    // SETUP 
    int length = solver->size;
    float r_norm;   // Residue norm
    float epsilon = 1e-5;  // Convergence threshold
    int max_iter = length;   // Maximum iterations
    float alpha;    // Step size along the search direction
    int k = 0;  // Iteration counter

    /*
    STEP ONE: Calculate initial residue r = b - Ax
    STEP TWO: Preconditioned residue z = D^(-1) * r
    where D is the diagonal of the matrix A
    */
    obm_matvec_mult(solver, &cl->x_buffer, &temp->Ap);

    update_r_and_z(solver, &cl->b_buffer, &temp->Ap, &temp->diagonal_buffer, 
            &temp->r_buffer, &temp->z_buffer, 1, length);

    /*
    STEP THREE: set first search direction p = z 
    */
    cl_event copy_buffer_evt;
    err = clEnqueueCopyBuffer(cl->q, temp->z_buffer, temp->p_buffer, 0, 0, 
        length * sizeof(float), 0, NULL, &copy_buffer_evt);
    ocl_check(err, "clEnqueueCopyBuffer failed for direction_buffer");

    /*
    STEP FOUR: main loop of the Conjugate Gradient algorithm
    */
    float r_dot_z = 0.0f;

    // Standard profiling 
    struct timespec start, end;
    struct timespec iter_start, iter_end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Advanced profiling
    cl_event wait_list[2];
    if(flags->profile) kernelstats_init(p, p->curr_iteration);

    VPRINTF(flags, "\n");
    do {
        clock_gettime(CLOCK_MONOTONIC, &iter_start);
        VPRINTF(flags, "\033[1;32mITERATION %d\033[0m\n", k);

        // Calculate alpha = dot(r, z) / dot(p, mat_vec(A, p))
        alpha = alpha_calculate(solver, &temp->r_buffer, &temp->z_buffer, 
                &temp->p_buffer, &temp->Ap, &r_dot_z, flags, p);
        VPRINTF(flags, "Alpha = %g\n", alpha);

        // Calculate the new residue r_(k+1) = r - alpha * mat_vec(A, p)
        // Calculate the new preconditioned residue z_(k+1) = D^(-1) * r_(k+1)
        wait_list[0] = update_r_and_z(solver, &temp->r_buffer, &temp->Ap, 
                &temp->diagonal_buffer, &temp->r_next_buffer, 
                &temp->z_next_buffer, alpha, length);
        
        // Calculate the norm of the new residue ||r_(k+1)||^2
        r_norm = dot_product_handler(solver, &temp->r_next_buffer, 
                &temp->r_next_buffer, length, flags, p);
        VPRINTF(flags, "Residue norm = %g\n", r_norm);

        // Calculate beta = dot(r_(k+1), z_(k+1)) / dot(r, z)
        float nextr_dot_nextz = dot_product_handler(solver, &temp->r_next_buffer, 
                &temp->z_next_buffer, length, flags, p); 
        float beta = nextr_dot_nextz / r_dot_z;
        r_dot_z = nextr_dot_nextz;
        VPRINTF(flags, "Beta = %g\n", beta);

        // Update the solution vector x = x + alpha * p
        // Update the search direction p = z_(k+1) + beta * p
        wait_list[1] = update_x_and_p(solver, &cl->x_buffer, 
                &temp->p_buffer, &temp->z_next_buffer, 
                &cl->x_buffer, &temp->p_buffer, alpha, beta, length);
        
        // Swap the buffers for the next iteration
        cl_mem tmp;
        tmp = temp->r_buffer;
        temp->r_buffer = temp->r_next_buffer;
        temp->r_next_buffer = tmp;

        tmp = temp->z_buffer;
        temp->z_buffer = temp->z_next_buffer;
        temp->z_next_buffer = tmp;

        // Advanced profiling
        if(flags->profile){
            clWaitForEvents(2, wait_list);
            
            profile_kernel(p, UPDATE_R_AND_Z, wait_list[0], p->curr_iteration);
            profile_kernel(p, UPDATE_X_AND_P, wait_list[1], p->curr_iteration);
        }     
        clReleaseEvent(wait_list[0]);
        clReleaseEvent(wait_list[1]);

        clock_gettime(CLOCK_MONOTONIC, &iter_end);
        float iter_elapsed = (iter_end.tv_sec - iter_start.tv_sec) + (iter_end.tv_nsec - iter_start.tv_nsec) / 1e9;
        VPRINTF(flags, "Iteration %d time: %.3f s\n", k, iter_elapsed);

        k++;

    } while(r_norm > epsilon && k < max_iter);

    clock_gettime(CLOCK_MONOTONIC, &end);
    float elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Advanced profiling
    if(flags->profile){
        p->cg_iterations[p->curr_iteration] = k;
        p->cg_elapsed[p->curr_iteration] = elapsed;
    }

    // Show the total energy (summation of each x_buffer value) 
    // at the end of the PCG simulation if the flag is set
    if(flags->show_energy) show_energy(solver, flags, p);

    VPRINTF(flags, "\033[1;32mCONVERGENCE\033[0m\n");
    VPRINTF(flags, "Iterations: %d\nNorm %g\n", k, r_norm);
    VPRINTF(flags, "Total time: %.3f s\n", elapsed);

    clFinish(cl->q);

    return elapsed;
}

float alpha_calculate(Solver* solver, cl_mem *r, cl_mem *z, cl_mem *p, cl_mem* Ap, float *r_dot_z, Flags *flags, Profiler *profiler) {
    int length = solver->size;

    // r * z 
    if (*r_dot_z == 0) 
        *r_dot_z = dot_product_handler(solver, r, z, length, flags, profiler);

    // p * A * p
    cl_event obm_matvec_evt = obm_matvec_mult(solver, p, Ap);
    
    if(flags->profile) {
        clWaitForEvents(1, &obm_matvec_evt);
        profile_kernel(profiler, OBM_MATVEC_MULT, obm_matvec_evt, profiler->curr_iteration);
    }
    clReleaseEvent(obm_matvec_evt);

    float denominator = dot_product_handler(solver, p, Ap, length, flags, profiler);
    if (denominator == 0) {
        fprintf(stderr, "Denominator is zero, cannot compute alpha.\n");
        exit(EXIT_FAILURE);
    }

    return *r_dot_z / denominator;
}

float dot_product_handler(Solver *solver, cl_mem *vec1, cl_mem *vec2, int length, Flags *flags, Profiler *p) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;

    if (length % 4 != 0) {
        fprintf(stderr, "dot_product_handler: length (%d) not multiple of 4 - unsupported.\n", length);
        exit(EXIT_FAILURE);
    }

    int length_vec4 = length / 4;                
    size_t num_groups = round_div_up(length_vec4, cl->lws);

    cl_mem partial_dot_product = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, sizeof(float) * length_vec4, NULL, &err);
    ocl_check(err, "clCreateBuffer failed for partial_dot_product");

    cl_event dot_event = dot_product_vec4(solver, vec1, vec2, &partial_dot_product, length_vec4);

    if(flags->profile) {
        clWaitForEvents(1, &dot_event);
        profile_kernel(p, DOT_PRODUCT_VEC4, dot_event, p->curr_iteration);
    }
    clReleaseEvent(dot_event);

    cl_mem temp_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, sizeof(float) * num_groups, NULL, &err);
    ocl_check(err, "clCreateBuffer failed for temp_buffer");

    cl_mem in_buf  = partial_dot_product;
    cl_mem out_buf = temp_buffer;

    float total_time = 0.0f;

    int elems_to_reduce = length_vec4; 
    while (elems_to_reduce > 1) {
        cl_event evt = reduce_sum4_float4_sliding(solver, &in_buf, &out_buf, elems_to_reduce);

        if (flags->profile) {
            clWaitForEvents(1, &evt);
            total_time += get_kernel_time(evt);
        }
        clReleaseEvent(evt);
        

        int n_vec4 = (elems_to_reduce + 3) / 4;
        int next_num_groups = (int)round_div_up((size_t)n_vec4, cl->lws);
        elems_to_reduce = next_num_groups;

        cl_mem tmp = in_buf;
        in_buf = out_buf;
        out_buf = tmp;
    }

    if(flags->profile) 
        add_kernel_sample(p, REDUCE_SUM4_FLOAT4_SLIDING, total_time, p->curr_iteration);

    float final_result = 0.0f;
    clEnqueueReadBuffer(cl->q, in_buf, CL_TRUE, 0, sizeof(float), &final_result, 0, NULL, NULL);

    clReleaseMemObject(partial_dot_product);
    clReleaseMemObject(temp_buffer);

    return final_result;
}

cl_event reduce_sum4_float4_sliding(Solver *solver, cl_mem* vec_in, cl_mem* vec_out, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Use the correct kernel (was incorrectly using partial_sum_reduction)
    err = clSetKernelArg(cl->kernels.reduce_sum4_float4_sliding, arg, sizeof(cl_mem), vec_in);
    ocl_check(err, "clSetKernelArg failed for vec_in");
    arg++;

    err = clSetKernelArg(cl->kernels.reduce_sum4_float4_sliding, arg, sizeof(cl_mem), vec_out);
    ocl_check(err, "clSetKernelArg failed for vec_out");
    arg++;

    // local memory: kernel reduces float4 elements -> allocate 4 floats per work-item
    err = clSetKernelArg(cl->kernels.reduce_sum4_float4_sliding, arg, cl->lws * 4 * sizeof(float), NULL);
    ocl_check(err, "clSetKernelArg failed for local_memory");
    arg++;

    err = clSetKernelArg(cl->kernels.reduce_sum4_float4_sliding, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for length_vec4");
    arg++;

    //Launch the kernel
    cl_event event;
    size_t n_vec4 = (length + 3) / 4;
    size_t gws = round_mul_up(n_vec4, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.reduce_sum4_float4_sliding, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for reduce_sum4_float4_sliding");

    return event;
}

cl_event dot_product_vec4(Solver *solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    int arg = 0;

    // length here is number of float4 elements
    err = clSetKernelArg(cl->kernels.dot_product_vec4, arg++, sizeof(cl_mem), vec1);
    ocl_check(err, "clSetKernelArg failed for vec1 (dot_product_vec4)");

    err = clSetKernelArg(cl->kernels.dot_product_vec4, arg++, sizeof(cl_mem), vec2);
    ocl_check(err, "clSetKernelArg failed for vec2 (dot_product_vec4)");
    
    err = clSetKernelArg(cl->kernels.dot_product_vec4, arg++, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result (dot_product_vec4)");

    // pass number of float4 elements (no local memory arg in kernel)
    err = clSetKernelArg(cl->kernels.dot_product_vec4, arg++, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for length (dot_product_vec4)");

    size_t gws = round_mul_up((size_t)length, cl->lws);
    cl_event event;
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.dot_product_vec4, 1, NULL, &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for dot_product_vec4");

    return event;
}

cl_event update_r_and_z(Solver* solver, cl_mem* r, cl_mem* Ap, cl_mem* precond, cl_mem* r_next, cl_mem* z_next, float alpha, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.update_r_and_z, arg, sizeof(cl_mem), r);
    ocl_check(err, "clSetKernelArg failed for r");
    arg++;

    err = clSetKernelArg(cl->kernels.update_r_and_z, arg, sizeof(cl_mem), Ap);
    ocl_check(err, "clSetKernelArg failed for Ap");
    arg++;

    err = clSetKernelArg(cl->kernels.update_r_and_z, arg, sizeof(cl_mem), precond);
    ocl_check(err, "clSetKernelArg failed for precond");
    arg++;

    err = clSetKernelArg(cl->kernels.update_r_and_z, arg, sizeof(cl_mem), r_next);
    ocl_check(err, "clSetKernelArg failed for r_next");
    arg++;

    err = clSetKernelArg(cl->kernels.update_r_and_z, arg, sizeof(cl_mem), z_next);
    ocl_check(err, "clSetKernelArg failed for z_next");
    arg++;

    err = clSetKernelArg(cl->kernels.update_r_and_z, arg, sizeof(float), &alpha);
    ocl_check(err, "clSetKernelArg failed for alpha");
    arg++;

    err = clSetKernelArg(cl->kernels.update_r_and_z, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for length");
    arg++;

    // Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.update_r_and_z, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for update_r");

    return event;
}

cl_event update_x_and_p(Solver* solver, cl_mem* x, cl_mem* p, cl_mem* z, cl_mem* x_next, cl_mem* p_next, float alpha, float beta, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.update_x_and_p, arg, sizeof(cl_mem), x);
    ocl_check(err, "clSetKernelArg failed for x");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x_and_p, arg, sizeof(cl_mem), p);
    ocl_check(err, "clSetKernelArg failed for p");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x_and_p, arg, sizeof(cl_mem), z);
    ocl_check(err, "clSetKernelArg failed for z");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x_and_p, arg, sizeof(cl_mem), x_next);
    ocl_check(err, "clSetKernelArg failed for x_next");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x_and_p, arg, sizeof(cl_mem), p_next);
    ocl_check(err, "clSetKernelArg failed for p_next");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x_and_p, arg, sizeof(float), &alpha);
    ocl_check(err, "clSetKernelArg failed for alpha");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x_and_p, arg, sizeof(float), &beta);
    ocl_check(err, "clSetKernelArg failed for beta");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x_and_p, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for length");
    arg++;

    // Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.update_x_and_p, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for update_x_and_p");

    return event;
}

cl_event obm_matvec_mult(Solver* solver, cl_mem* vec, cl_mem* result) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.obm_matvec_mult, arg, sizeof(cl_mem), &cl->obm_values_buffer);
    ocl_check(err, "clSetKernelArg failed for obm_values_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.obm_matvec_mult, arg, sizeof(cl_mem), &cl->obm_offset_buffer);
    ocl_check(err, "clSetKernelArg failed for obm_offset_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.obm_matvec_mult, arg, sizeof(int), &solver->A_obm.non_zero_values);
    ocl_check(err, "clSetKernelArg failed for non_zeros_values");
    arg++;

    err = clSetKernelArg(cl->kernels.obm_matvec_mult, arg, sizeof(cl_mem), vec);
    ocl_check(err, "clSetKernelArg failed for vec");
    arg++;

    err = clSetKernelArg(cl->kernels.obm_matvec_mult, arg, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl->kernels.obm_matvec_mult, arg, sizeof(int), &solver->size);
    ocl_check(err, "clSetKernelArg failed for size");
    arg++;

    //Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(solver->size, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.obm_matvec_mult, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for obm_matvec_mult");

    return event;
}

cl_event obm_matvec_mult_local(Solver* solver, cl_mem* vec, cl_mem* result) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    int max_offset = solver->A_obm.offset[solver->A_obm.non_zero_values - 1];
    int local_mem_size = cl->lws + 2*max_offset; 
    local_mem_size *= sizeof(float); 

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.obm_matvec_mult_local, arg, sizeof(cl_mem), &cl->obm_values_buffer);
    ocl_check(err, "clSetKernelArg local failed for obm_values_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.obm_matvec_mult_local, arg, sizeof(cl_mem), &cl->obm_offset_buffer);
    ocl_check(err, "clSetKernelArg failed for obm_offset_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.obm_matvec_mult_local, arg, sizeof(int), &solver->A_obm.non_zero_values);
    ocl_check(err, "clSetKernelArg failed for non_zeros_values");
    arg++;

    err = clSetKernelArg(cl->kernels.obm_matvec_mult_local, arg, local_mem_size, NULL);
    ocl_check(err, "clSetKernelArg failed for local_memory");
    arg++;

    err = clSetKernelArg(cl->kernels.obm_matvec_mult_local, arg, sizeof(cl_mem), vec);
    ocl_check(err, "clSetKernelArg failed for vec");
    arg++;

    err = clSetKernelArg(cl->kernels.obm_matvec_mult_local, arg, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl->kernels.obm_matvec_mult_local, arg, sizeof(int), &solver->size);
    ocl_check(err, "clSetKernelArg failed for size");
    arg++;

    //Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(solver->size, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.obm_matvec_mult_local, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for obm_matvec_mult_local");

    return event;
}

void save_result(Solver *solver, size_t size, float* result) {
    OpenCLContext *cl = &solver->cl;

    cl_int err = clEnqueueReadBuffer(cl->q, 
            cl->x_buffer, CL_TRUE, 0, size * sizeof(float), result, 0, NULL, NULL);
    ocl_check(err, "print_buffer read");
}

void show_energy(Solver *solver, Flags *flags, Profiler *p){
    cl_int err;

    float *ones = malloc(solver->size * sizeof(float));
    for(int i = 0; i < solver->size; i++) ones[i] = 1.0f;

    cl_mem ones_buffer = clCreateBuffer(solver->cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
            solver->size * sizeof(float), ones, &err);
    
    float energy = dot_product_handler(solver, &solver->cl.x_buffer, 
            &ones_buffer, solver->size, flags, p);
    printf("\nTotal energy: %.0f", energy);

    clReleaseMemObject(ones_buffer);
    free(ones);
}

void free_cg_solver(Solver* solver) {
    OpenCLContext* cl = &solver->cl;
    
    if(solver->x) free(solver->x);
    if(solver->b) free(solver->b);

    if(cl->b_buffer) clReleaseMemObject(cl->b_buffer);
    if(cl->x_buffer) clReleaseMemObject(cl->x_buffer);

    if(cl->obm_offset_buffer) clReleaseMemObject(cl->obm_offset_buffer);
    if(cl->obm_values_buffer) clReleaseMemObject(cl->obm_values_buffer);
    if(cl->kernels.dot_product_vec4) clReleaseKernel(cl->kernels.dot_product_vec4);
    if(cl->kernels.reduce_sum4_float4_sliding) clReleaseKernel(cl->kernels.reduce_sum4_float4_sliding);
    if(cl->kernels.update_r_and_z) clReleaseKernel(cl->kernels.update_r_and_z);
    if(cl->kernels.update_x_and_p) clReleaseKernel(cl->kernels.update_x_and_p);
    if(cl->kernels.obm_matvec_mult) clReleaseKernel(cl->kernels.obm_matvec_mult);
    if(cl->kernels.obm_matvec_mult_local) clReleaseKernel(cl->kernels.obm_matvec_mult_local);

    free_temporary_buffers(cl);
        
    if(cl->prog) clReleaseProgram(cl->prog);
    if(cl->q) clReleaseCommandQueue(cl->q);
    if(cl->ctx) clReleaseContext(cl->ctx);
} 

void free_temporary_buffers(OpenCLContext *cl){
    if(cl->temp.Ap) clReleaseMemObject(cl->temp.Ap);
    if(cl->temp.diagonal_buffer) clReleaseMemObject(cl->temp.diagonal_buffer);
    if(cl->temp.r_buffer) clReleaseMemObject(cl->temp.r_buffer);
    if(cl->temp.z_buffer) clReleaseMemObject(cl->temp.z_buffer);
    if(cl->temp.p_buffer) clReleaseMemObject(cl->temp.p_buffer);
    if(cl->temp.r_next_buffer) clReleaseMemObject(cl->temp.r_next_buffer);
    if(cl->temp.z_next_buffer) clReleaseMemObject(cl->temp.z_next_buffer);
}

// SUPPORT FUNCTIONS FOR THE CRANK NICOLSON SOLVER, THEY ARE NOT CONJUGATE GRADIENT STEPS
void update_unknown(Solver *solver){
    OpenCLContext cl = solver->cl;
    cl_int err;
    cl_int arg = 0;

    err = clSetKernelArg(cl.kernels.obm_matvec_mult, arg, sizeof(cl_mem), &cl.B_obm_values_buffer);
    ocl_check(err, "clSetKernelArg failed for obm_values_buffer");
    arg++;

    err = clSetKernelArg(cl.kernels.obm_matvec_mult, arg, sizeof(cl_mem), &cl.obm_offset_buffer);
    ocl_check(err, "clSetKernelArg failed for obm_offset_buffer");
    arg++;

    err = clSetKernelArg(cl.kernels.obm_matvec_mult, arg, sizeof(int), &solver->A_obm.non_zero_values);
    ocl_check(err, "clSetKernelArg failed for non_zeros_values");
    arg++;

    err = clSetKernelArg(cl.kernels.obm_matvec_mult, arg, sizeof(cl_mem), &cl.x_buffer);
    ocl_check(err, "clSetKernelArg failed for vec");
    arg++;

    err = clSetKernelArg(cl.kernels.obm_matvec_mult, arg, sizeof(cl_mem), &cl.b_buffer);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl.kernels.obm_matvec_mult, arg, sizeof(int), &solver->size);
    ocl_check(err, "clSetKernelArg failed for size");
    arg++;
    
    cl_event event;
    size_t gws = round_mul_up(solver->size, cl.lws);
    err = clEnqueueNDRangeKernel(cl.q, cl.kernels.obm_matvec_mult, 1, NULL,
            &gws, &cl.lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for obm_matvec_mult");
}

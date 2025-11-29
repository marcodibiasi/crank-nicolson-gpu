#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <time.h>
#include "conjugate_gradient.h"
#include "ocl_boiler.h"
#include "flags.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif


Solver setup_solver(int size, OBMatrix A_obm, float *b, float *initial_x) {
    Solver solver;
    solver.size = size;
    solver.A_obm = A_obm;
    solver.b = b;
    solver.x = initial_x;

    if (!solver.x) {
        solver.x = malloc(size * sizeof(float));
        if (!solver.x) {
            fprintf(stderr, "Failed to allocate memory for x vector\n");
            exit(1);
        }
        for (int i = 0; i < size; i++) {
            solver.x[i] = 0.0;
        }
    }
    solver.cl = setup_opencl_context(&solver);  

    return solver;
}

OpenCLContext setup_opencl_context(Solver* solver) {
    OpenCLContext cl;

    cl.p = select_platform();
	cl.d = select_device(cl.p);
	cl.ctx = create_context(cl.p, cl.d); 
	cl.q = create_queue(cl.ctx, cl.d);
	cl.prog = create_program("../ConjugateGradient/kernels.cl", cl.ctx, cl.d);

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
        solver->A_obm.non_zero_values * sizeof(float), solver->A_obm.offset, &err);
    cl.obm_values_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        solver->A_obm.non_zero_values * sizeof(float), solver->A_obm.values, &err);
    
    // Create kernels
    cl.kernels.reduce_sum4_float4_sliding = clCreateKernel(cl.prog, "reduce_sum4_float4_sliding", &err);
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.dot_product_vec4 = clCreateKernel(cl.prog, "dot_product_vec4", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.obm_matvec_mult = clCreateKernel(cl.prog, "obm_matvec_mult", &err);
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.update_x = clCreateKernel(cl.prog, "update_x", &err);
    ocl_check(err, "clCreateKernel failed for update_x");

    cl.kernels.update_r_and_z = clCreateKernel(cl.prog, "update_r_and_z", &err);
    ocl_check(err, "clCreateKernel failed for update_r");

    cl.kernels.update_p = clCreateKernel(cl.prog, "update_p", &err);
    ocl_check(err, "clCreateKernel failed for update_p");

    cl.lws = 128;

    return cl;
}

float conjugate_gradient(Solver* solver, Flags *flags) {
    cl_int err;
    OpenCLContext* cl = &solver->cl;

    // SETUP 
    int length = solver->size;
    float r_norm;   // Residue norm
    float epsilon = 1e-5;  // Convergence threshold
    int max_iter = length;   // Maximum iterations
    float alpha;    // Step size along the search direction
    int k = 0;  // Iteration counter

    /*
    STEP ZERO: Precondition with Jacobi by extracting the diagonal of the matrix A
    and inverting it
    */
    cl_mem diagonal_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err); 

    //symmetrical offsets
    float diagonal_value =1.0f / solver->A_obm.values[solver->A_obm.non_zero_values/2];
    ocl_check(err, "clCreateBuffer failed for diagonal_buffer");
    err = clEnqueueFillBuffer(cl->q, diagonal_buffer, &diagonal_value, sizeof(float), 0,
                    sizeof(float) * length, 0, NULL, NULL);
    ocl_check(err, "clEnqueueFillBuffer failed for diagonal_buffer");

    /*
    STEP ONE: Calculate initial residue r = b - Ax
    STEP TWO: Preconditioned residue z = D^(-1) * r
    where D is the diagonal of the matrix A
    */
    cl_mem r_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for r_buffer");

    cl_mem Ap = clCreateBuffer(solver->cl.ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for Ap");

    cl_event mat_vec_multiply_evt = obm_matvec_mult(solver, &cl->x_buffer, &Ap);
    clWaitForEvents(1, &mat_vec_multiply_evt);
    clReleaseEvent(mat_vec_multiply_evt);

    cl_mem z_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for z_buffer");

    cl_event initial_r_z = update_r_and_z(solver, &cl->b_buffer, &Ap, &diagonal_buffer, &r_buffer, &z_buffer, 1, length);

    /*
    STEP THREE: set first search direction p = z 
    */
    cl_mem direction_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
	ocl_check(err, "clCreateBuffer failed for direction_buffer");

    cl_event copy_buffer_evt;
    err = clEnqueueCopyBuffer(cl->q, z_buffer, direction_buffer, 0, 0, 
        length * sizeof(float), 0, NULL, &copy_buffer_evt);
    ocl_check(err, "clEnqueueCopyBuffer failed for direction_buffer");

    clWaitForEvents(1, &copy_buffer_evt);
    clReleaseEvent(copy_buffer_evt);

    /*
    STEP FOUR: main loop of the Conjugate Gradient algorithm
    */
    // Allocate buffers for the next iteration
    cl_mem r_next_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for r_next_buffer");

    cl_mem z_next_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for r_next_buffer");

    float r_dot_z = 0.0f;

    // profiling 
    struct timespec start, end;
    struct timespec iter_start, iter_end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    do {
        clock_gettime(CLOCK_MONOTONIC, &iter_start);

        VPRINTF(flags, "\033[1;32mITERATION %d\033[0m\n", k);
        // alpha = dot(r, z) / dot(p, mat_vec(A, p))
        VPRINTF(flags, "ALPHA CALCULATE\n");
        float alpha = alpha_calculate(solver, &r_buffer, &z_buffer, &direction_buffer, &Ap, &r_dot_z, flags);
        VPRINTF(flags, "\n\tAlpha = %g\n", alpha);

        // Update the solution vector x = x + alpha * p
        VPRINTF(flags, "\nUPDATE X\n");
        cl_event upd_x = update_x(solver, &direction_buffer, alpha, length);
        
        if(flags->profile) {
            clWaitForEvents(1, &upd_x);
            float updx_t = profiling_event(upd_x);
            printf("%-40s %-6.3f ms\n", "\tupdate_x kernel:", updx_t);
            clReleaseEvent(upd_x);
            float gbps_update_x = (sizeof(float) * 3.0 * length) / (updx_t * 1e6);
            printf("%-40s %.3f GB/s\n", "\tUpdate x speed:", gbps_update_x);
        }

        // Calculate the new residue r_(k+1) = r - alpha * mat_vec(A, p)
        // Calculate the new preconditioned residue z_(k+1) = D^(-1) * r_(k+1)
        VPRINTF(flags, "\nUPDATE R AND Z\n");
        cl_event next_r_z = update_r_and_z(solver, &r_buffer, &Ap, &diagonal_buffer, &r_next_buffer, &z_next_buffer, alpha, length);
        if(flags->profile) {
            clWaitForEvents(1, &next_r_z);
            float nextrz_t = profiling_event(next_r_z);
            printf("%-40s %-6.3f ms\n", "\tupdate_r_and_z kernel:", nextrz_t);
            clReleaseEvent(next_r_z);
            float gbps_update_rz = (5 * length * sizeof(float)) / (nextrz_t * 1e6);
            printf("%-40s %.3f GB/s\n", "\tUpdate r and z speed:", gbps_update_rz);
        }

        VPRINTF(flags, "\nCALCULATE ||r_(k+1)||^2\n");    
        // Calculate the norm of the new residue ||r_(k+1)||^2
        r_norm = dot_product_handler(solver, &r_next_buffer, &r_next_buffer, length, flags);
        VPRINTF(flags, "\n\tResidue norm = %g\n", r_norm);


        // beta = dot(r_(k+1), z_(k+1)) / dot(r, z)
        VPRINTF(flags, "\nBETA CALCULATE\n");
        VPRINTF(flags, "\t(r_(k+1) · z_(k+1))\n");
        float nextr_dot_nextz = dot_product_handler(solver, &r_next_buffer, &z_next_buffer, solver->size, flags); 
        float beta = nextr_dot_nextz / r_dot_z;
        r_dot_z = nextr_dot_nextz;
        VPRINTF(flags, "\n\tBeta = %g\n", beta);

        // Update the search direction p = z_(k+1) + beta * p
        VPRINTF(flags, "\nUPDATE P\n");
        cl_event upd_p = update_p(solver, &direction_buffer, &z_next_buffer, beta, length);
        if(flags->profile) {
            clWaitForEvents(1, &upd_p);
            float updp_t = profiling_event(upd_p);
            printf("%-40s %-6.3f ms\n", "\tupdate_p kernel:", updp_t);
            clReleaseEvent(upd_p);
            float gbps_update_p = (sizeof(float) * 3.0 * length) / (updp_t * 1e6);
            printf("%-40s %.3f GB/s\n", "\tUpdate p speed:", gbps_update_p);
        }

        // Swap the buffers for the next iteration
        cl_mem tmp;
        tmp = r_buffer;
        r_buffer = r_next_buffer;
        r_next_buffer = tmp;

        tmp = z_buffer;
        z_buffer = z_next_buffer;
        z_next_buffer = tmp;


        clock_gettime(CLOCK_MONOTONIC, &iter_end);
        float iter_elapsed = (iter_end.tv_sec - iter_start.tv_sec) + (iter_end.tv_nsec - iter_start.tv_nsec) / 1e9;
        VPRINTF(flags, "\nIteration %d time: %.3f s\n", k, iter_elapsed);

        VPRINTF(flags, "\n");
        k++;

    } while(r_norm > epsilon && k < max_iter);

    clock_gettime(CLOCK_MONOTONIC, &end);
    float elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Show the total energy for at the end of the PCG simulation if the flag is set
    if(flags->show_energy) {
        float *ones = malloc(solver->size * sizeof(float));
        for (int i = 0; i < solver->size; i++) ones[i] = 1.0f;

        cl_mem ones_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            solver->size * sizeof(float), ones, &err);
        
        float energy = dot_product_handler(solver, &cl->x_buffer, &ones_buffer, solver->size, flags);
        printf("\nTotal energy: %.0f", energy);

        clReleaseMemObject(ones_buffer);
        free(ones);
    }

    save_result(solver, cl->x_buffer, length * sizeof(float), 0);

    clReleaseMemObject(Ap);
    clReleaseMemObject(diagonal_buffer);
    clReleaseMemObject(r_buffer);
    clReleaseMemObject(z_buffer);
    clReleaseMemObject(direction_buffer);
    clReleaseMemObject(r_next_buffer);
    clReleaseMemObject(z_next_buffer);

    VPRINTF(flags, "\nIterations: %d\nNorm %g\n", k, r_norm);
    VPRINTF(flags, "Total time: %.3f s\n", elapsed);

    clFinish(cl->q);

    return elapsed;
}

float alpha_calculate(Solver* solver, cl_mem *r, cl_mem *z, cl_mem *p, cl_mem* Ap, float *r_dot_z, Flags *flags) {
    cl_int err;
    OpenCLContext *cl = &solver->cl;
    int length = solver->size;

    // r * z 
    if (*r_dot_z == 0) {
        VPRINTF(flags, "\t(r · z)\n");
        *r_dot_z = dot_product_handler(solver, r, z, length, flags);
    }

    // p * A * p
    VPRINTF(flags, "\n\t(p * A * p)\n");
    cl_event mat_vec_multiply_evt = obm_matvec_mult(solver, p, Ap);
    
    if(flags->profile) {
        clWaitForEvents(1, &mat_vec_multiply_evt);
        float t = profiling_event(mat_vec_multiply_evt);
        printf("%-40s %-6.3f ms\n", "\tmat_vec_multiply kernel:", t);
        clReleaseEvent(mat_vec_multiply_evt);
        float bytes =
            (solver->A_obm.non_zero_values * (sizeof(float) + sizeof(int) + sizeof(float))
            + sizeof(float)) * length;
        float gbps = bytes / (t * 1e6);
        printf("%-40s %.3f GB/s\n", "\tMat vec multiply speed:", gbps);
    }

    float denominator = dot_product_handler(solver, p, Ap, length, flags);

    // printf("Alpha -> Numerator: %g, Denominator: %g\n", numerator, denominator);

    if (denominator == 0) {
        fprintf(stderr, "Denominator is zero, cannot compute alpha.\n");
        exit(EXIT_FAILURE);
    }

    return *r_dot_z / denominator;
}

float dot_product_handler(Solver *solver, cl_mem *vec1, cl_mem *vec2, int length, Flags *flags) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;

    if (length % 4 != 0) {
        fprintf(stderr, "dot_product_handler: length (%d) not multiple of 4 — unsupported.\n", length);
        exit(EXIT_FAILURE);
    }

    int length_vec4 = length / 4;                
    size_t num_groups = round_div_up(length_vec4, cl->lws);

    cl_mem partial_dot_product = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, sizeof(float) * length_vec4, NULL, &err);
    ocl_check(err, "clCreateBuffer failed for partial_dot_product");

    cl_event dot_event = dot_product_vec4(solver, vec1, vec2, &partial_dot_product, length_vec4);

    if(flags->profile) {
        clWaitForEvents(1, &dot_event);
        float t = profiling_event(dot_event);
        printf("%-40s %-6.3f ms\n", "\tdot_product_vec4 kernel:", t);
        clReleaseEvent(dot_event);
        float gbps = (sizeof(float) * 2 * length) / (t * 1e6);
        printf("%-40s %.3f GB/s\n", "\tDot product speed:", gbps);
    }

    cl_mem temp_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, sizeof(float) * num_groups, NULL, &err);
    ocl_check(err, "clCreateBuffer failed for temp_buffer");

    cl_mem in_buf  = partial_dot_product;
    cl_mem out_buf = temp_buffer;

    float total_time = 0.0f;
    float total_bytes = 0.0f;

    int elems_to_reduce = length_vec4; 
    while (elems_to_reduce > 1) {
        cl_event evt = reduce_sum4_float4_sliding(solver, &in_buf, &out_buf, elems_to_reduce);

        if (flags->profile) {
            clWaitForEvents(1, &evt);
            total_time += profiling_event(evt);
            clReleaseEvent(evt);
        }

        total_bytes += elems_to_reduce * sizeof(float);

        int n_vec4 = (elems_to_reduce + 3) / 4;
        int next_num_groups = (int)round_div_up((size_t)n_vec4, cl->lws);
        elems_to_reduce = next_num_groups;

        cl_mem tmp = in_buf;
        in_buf = out_buf;
        out_buf = tmp;
    }

    if (flags->profile) {
        printf("%-40s %-6.3f ms\n", "\treduce_sum4_float4_sliding (total):", total_time);
        if (total_time > 0.0f)
            printf("%-40s %.3f GB/s\n", "\treduction speed:", total_bytes / (total_time * 1e6));
    }

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

cl_event update_x(Solver* solver, cl_mem* p, float alpha, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.update_x, arg, sizeof(cl_mem), &cl->x_buffer);
    ocl_check(err, "clSetKernelArg failed for x_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x, arg, sizeof(cl_mem), p);
    ocl_check(err, "clSetKernelArg failed for p");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x, arg, sizeof(float), &alpha);
    ocl_check(err, "clSetKernelArg failed for alpha");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for length");
    arg++;

    // Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.update_x, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for update_x");

    return event;
} 

cl_event update_p(Solver* solver, cl_mem* p, cl_mem* z, float beta, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.update_p, arg, sizeof(cl_mem), p);
    ocl_check(err, "clSetKernelArg failed for p");
    arg++;

    err = clSetKernelArg(cl->kernels.update_p, arg, sizeof(cl_mem), z);
    ocl_check(err, "clSetKernelArg failed for z");
    arg++;

    err = clSetKernelArg(cl->kernels.update_p, arg, sizeof(float), &beta);
    ocl_check(err, "clSetKernelArg failed for beta");
    arg++;

    err = clSetKernelArg(cl->kernels.update_p, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for length");
    arg++;

    // Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.update_p, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for update_p");

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

void free_cg_solver(Solver* solver) {
    OpenCLContext* cl = &solver->cl;

    if(solver->x) free(solver->x);

    if(cl->b_buffer) clReleaseMemObject(cl->b_buffer);
    if(cl->x_buffer) clReleaseMemObject(cl->x_buffer);

    if(cl->obm_offset_buffer) clReleaseMemObject(cl->obm_offset_buffer);
    if(cl->obm_values_buffer) clReleaseMemObject(cl->obm_values_buffer);
    if(cl->kernels.dot_product_vec4) clReleaseKernel(cl->kernels.dot_product_vec4);
    if(cl->kernels.reduce_sum4_float4_sliding) clReleaseKernel(cl->kernels.reduce_sum4_float4_sliding);
    if(cl->kernels.update_x) clReleaseKernel(cl->kernels.update_x);
    if(cl->kernels.update_p) clReleaseKernel(cl->kernels.update_p);
    if(cl->kernels.update_r_and_z) clReleaseKernel(cl->kernels.update_r_and_z);
    if(cl->kernels.obm_matvec_mult) clReleaseKernel(cl->kernels.obm_matvec_mult);

    if(cl->prog) clReleaseProgram(cl->prog);
    if(cl->q) clReleaseCommandQueue(cl->q);
    if(cl->ctx) clReleaseContext(cl->ctx);
}  

void save_result(Solver *solver, cl_mem buf, size_t size, int n) {
    OpenCLContext *cl = &solver->cl;
    float* temp = malloc(size);
    if (!temp) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }

    clFinish(cl->q); 
    cl_int err = clEnqueueReadBuffer(cl->q, buf, CL_TRUE, 0, size, solver->x, 0, NULL, NULL);
    ocl_check(err, "print_buffer read");

    for (int i = 0; i < n; i++) {
        printf(" %.2f ", solver->x[i]);
    }
    // printf("\n");
    free(temp);
}

float profiling_event(cl_event event) {
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    return (end - start) / 1e6; // Convert to milliseconds
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
    size_t gws = solver->size * cl->lws;
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.obm_matvec_mult, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for obm_matvec_mult");

    return event;
}

// SUPPORT FUNCTIONS FOR THE CRANK NICOLSON SOLVER, THEY ARE NOT CONJUGATE GRADIENT STEPS
float* calculate_unknown_vector(OpenCLContext cl, OBMatrix B, float* u_n) {
    cl_int err;
    cl_int arg = 0;

    cl_mem offset = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        B.non_zero_values * sizeof(float), B.offset, &err);
    cl_mem values = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        B.non_zero_values * sizeof(float), B.values, &err);
    cl_mem vec = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        B.rows * sizeof(float), u_n, &err);
    cl_mem result = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, B.rows * sizeof(float), NULL, &err);

    err = clSetKernelArg(cl.kernels.obm_matvec_mult, arg, sizeof(cl_mem), &values);
    ocl_check(err, "clSetKernelArg failed for obm_values_buffer");
    arg++;

    err = clSetKernelArg(cl.kernels.obm_matvec_mult, arg, sizeof(cl_mem), &offset);
    ocl_check(err, "clSetKernelArg failed for obm_offset_buffer");
    arg++;

    err = clSetKernelArg(cl.kernels.obm_matvec_mult, arg, sizeof(int), &B.non_zero_values);
    ocl_check(err, "clSetKernelArg failed for non_zeros_values");
    arg++;

    err = clSetKernelArg(cl.kernels.obm_matvec_mult, arg, sizeof(cl_mem), &vec);
    ocl_check(err, "clSetKernelArg failed for vec");
    arg++;

    err = clSetKernelArg(cl.kernels.obm_matvec_mult, arg, sizeof(cl_mem), &result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl.kernels.obm_matvec_mult, arg, sizeof(int), &B.rows);
    ocl_check(err, "clSetKernelArg failed for size");
    arg++;
    
    cl_event event;
    size_t gws = B.rows * cl.lws;
    err = clEnqueueNDRangeKernel(cl.q, cl.kernels.obm_matvec_mult, 1, NULL,
            &gws, &cl.lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for obm_matvec_mult");

    float *b = malloc(B.rows * sizeof(float));
    err = clEnqueueReadBuffer(cl.q, result, CL_TRUE, 0, B.rows * sizeof(float), b, 
            0, NULL, NULL);
    ocl_check(err, "clEnqueueReadBuffer failed for result");

    clReleaseMemObject(offset);
    clReleaseMemObject(values);
    clReleaseMemObject(vec);
    clReleaseMemObject(result);

    return b;
}

void update_unknown_b(Solver* solver, float* b){
    cl_int err;
    err = clEnqueueWriteBuffer(solver->cl.q, solver->cl.b_buffer, CL_TRUE, 0,
        solver->size * sizeof(float), b, 0, NULL, NULL);
    ocl_check(err, "clEnqueueWriteBuffer failed for update_b");
}
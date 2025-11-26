__kernel void reduce_sum4_float4_sliding(
    __global const float4* restrict in,
    __global float* restrict out,
    __local float* restrict local_memory,
    const int n_els  // total number
) {
    const int li = get_local_id(0);
    const int lws = get_local_size(0);
    const int group_id = get_group_id(0);
    const int num_groups = get_num_groups(0);

    const int nels_vec = (n_els + 3) / 4; 
    const int vecs_per_group_min = (nels_vec - 1) / num_groups + 1;
    const int vecs_per_group = lws * ((vecs_per_group_min - 1) / lws + 1);

    int gi = group_id * vecs_per_group + li;
    const int end = (group_id + 1) * vecs_per_group;

    float acc = 0.0;

    while (gi < end) {
        float val = 0.0;
        if (gi < nels_vec) {
            float4 d = in[gi];
            int base = gi * 4;
            if (base + 3 < n_els)
                val = d.x + d.y + d.z + d.w;
            else {
                if (base + 0 < n_els) val += d.x;
                if (base + 1 < n_els) val += d.y;
                if (base + 2 < n_els) val += d.z;
                if (base + 3 < n_els) val += d.w;
            }
        }

        local_memory[li] = val;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = lws / 2; stride > 0; stride >>= 1) {
            if (li < stride)
                local_memory[li] += local_memory[li + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (li == 0)
            acc += local_memory[0];

        gi += lws;
    }

    if (li == 0)
        out[group_id] = acc;
}


__kernel void dot_product_vec4(
    __global const float4* a,
    __global const float4* b,
    __global float* result,
    const int length 
){
    int global_id = get_global_id(0);
    if(global_id >= length) return;

    float local_product = 0.0;
    float4 a_val = a[global_id];
    float4 b_val = b[global_id];

    local_product += (a_val.x * b_val.x) + (a_val.y * b_val.y) +
        (a_val.z * b_val.z) + (a_val.w * b_val.w);

    result[global_id] = local_product;
}


__kernel void obm_matvec_mult(

    //Offsetted Matrix format
    __global const float* values,
    __global const int* offset,
    const int non_zeros,

    __global const float* vec_in,
    __global float* vec_out,

    const int rows
)
{
    int global_id = get_global_id(0);
    if (global_id >= rows) return;

    float sum = 0.0;

    for (int i = 0; i < non_zeros; i++) {
        int j = global_id + offset[i];
        if (j >= 0 && j < rows) {
            sum += values[i] * vec_in[j];
        }
    }

    vec_out[global_id] = sum;
}


__kernel void mult_vectors(
    __global const float* vec1,
    __global const float* vec2,
    __global float* result,
    const int lenght
)
{
    int global_id = get_global_id(0);
    if (global_id < lenght) 
        result[global_id] = vec1[global_id] * vec2[global_id];
}


__kernel void sum_vectors(
    __global const float* vec1,
    __global const float* vec2,
    __global float* result,
    const int lenght
)
{
    int global_id = get_global_id(0);
    if (global_id < lenght) 
        result[global_id] = vec1[global_id] + vec2[global_id];
}


__kernel void scale_vector(
    __global const float* vec_in,
    __global float* vec_out,
    const float scale,
    const int lenght
)
{
    int global_id = get_global_id(0);
    if (global_id < lenght) 
        vec_out[global_id] = vec_in[global_id] * scale;
}

__kernel void update_x(
    __global float* x,
    __global const float* p,
    const float alpha,
    const int length
)
{
    int global_id = get_global_id(0);
    if (global_id < length) 
        x[global_id] += alpha * p[global_id];
}  

__kernel void update_r_and_z(
    __global const float* r,
    __global const float* Ap,
    __global const float* precond,
    __global float* r_next,  // at the start it contains A * p  
    __global float* z_next,
    const float alpha,
    const int length
)
{
    int global_id = get_global_id(0);
    float r_i;

    if (global_id < length) {
        r_i = r[global_id] - alpha * Ap[global_id];
        r_next[global_id] = r_i;
        z_next[global_id] = r_i * precond[global_id];
    }
}  

__kernel void update_p(
    __global float* p,
    __global const float* z,
    const float beta,
    const int length
)
{
    int global_id = get_global_id(0);
    if (global_id < length) {     
        p[global_id] = z[global_id] + beta * p[global_id];
    }
} 
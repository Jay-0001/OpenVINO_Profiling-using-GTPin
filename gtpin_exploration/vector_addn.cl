__kernel void vector_add(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const int N)
{
    int gid = get_global_id(0);

    if (gid < N) {
        C[gid] = A[gid] + B[gid];
    }
}
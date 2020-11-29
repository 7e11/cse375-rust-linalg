/// This is what the add.ptx is compiled from
/// "nvcc add.cu --ptx"
extern "C" __global__ void sum(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] + y[i];
    }
}

/// From this PDF: https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/cuda/05-cuda-mm.pdf?__blob=publicationFile
/// Also, see this similar guide: https://www.tutorialspoint.com/cuda/cuda_matrix_multiplication.htm
/// It's called like so:
/// mm_kernel<<<dimGrid, dimBlock>>> (d_a, d_b, d_c, n);
/// Naive. Can this be done using shared memory?
/// Also, this accesses global memory (A and B) twice per loop. See: https://www.tutorialspoint.com/cuda/cuda_performance_considerations.htm
extern "C" __global__ void mm_kernel(float* A, float* B, float* C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;    // block Index * how wide the block is + thread index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        for (int i = 0; i < n; ++i) {
            C[row * n + col] += A[row * n + i] * B[i * n + col];
        }
    }

    // Multiplying IxJ matrix with JxK matrix gives a matrix with IxK
    // another way to think about this:
//    float product_val = 0
//    for(int k=0;k<width;k++) {
//      product_val += d_M[row*width+k]*d_N[k*width+col];
//    }
//    d_p[row*width+col] = product_val;
}
#include <iostream>
#include <vector>
#include <cmath>

/**
 * PROBLEM: Vector Addition (C = A + B)
 * GOAL: Understand the basic lifecycle of a CUDA program.
 * 
 * TODO: Define the __global__ kernel 'vectorAdd' below.
 * Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels
 */

// TODO: Write your kernel here
// __global__ void vectorAdd(...) { ... }

int main() {
    // Total number of elements
    const int N = 1 << 20; // 1,048,576 elements
    const size_t bytes = N * sizeof(float);

    // Host vectors (CPU memory)
    std::vector<float> h_A(N), h_B(N), h_C(N);

    // Initialize inputs
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    /**
     * STAGE 1: Memory Allocation (Silicon)
     * TODO: Allocate device memory for d_A, d_B, and d_C on the GPU.
     * Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#ga37d3796527d4c7711d99265180cabaae
     */
    float *d_A, *d_B, *d_C;
    // cudaMalloc(&d_A, ...);
    // ...

    /**
     * STAGE 2: Copy Data to GPU
     * TODO: Copy h_A and h_B from the CPU to the GPU.
     * Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#ga48efa57b3ca9623e595c25605d398d83
     */
    // cudaMemcpy(d_A, h_A.data(), ...);
    // ...

    /**
     * STAGE 3: Kernel Execution
     * TODO: Launch the 'vectorAdd' kernel.
     * Use a 1D grid with 256 threads per block. Calculate the number of blocks needed.
     * Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels
     */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(...);

    /**
     * STAGE 4: Copy Results Back
     * TODO: Copy the resulting d_C from the GPU back to h_C on the CPU.
     */
    // cudaMemcpy(h_C.data(), d_C, ...);

    // STAGE 5: Verification (Synapse)
    std::cout << "Verifying results..." << std::endl;
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            std::cerr << "Verification failed at index " << i << "!" << std::endl;
            std::cerr << "Expected: " << h_A[i] + h_B[i] << ", Got: " << h_C[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "SUCCESS! GPU and CPU results match." << std::endl;
    }

    /**
     * STAGE 6: Cleanup
     * TODO: Free the device memory for d_A, d_B, and d_C.
     */
    // cudaFree(d_A);
    // ...

    return 0;
}

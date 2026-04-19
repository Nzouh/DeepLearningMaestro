#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * PROBLEM: Matrix Multiplication (C = A * B)
 * GOAL: Understand 2D indexing and grid/block layout.
 * 
 * For simplicity, we will use square matrices of size N x N.
 * Matrix A: N rows x N cols
 * Matrix B: N rows x N cols
 * Matrix C: N rows x N cols
 */

/**
 * TODO: Define the __global__ kernel 'matrixMul' below.
 * Each thread should compute ONE element of the output matrix C.
 */
__global__ void matrixMul(const float *A, const float *B, float *C, int N) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(col < N && row < N){
        int k = 0;
        float sum = 0.0f;
        while (k < N){
            sum += A[row * N + k] * B[k * N + col];
            k++;
        }
        C[N * row + col] = sum;
    }


    // 1. Calculate the row and column index for this thread
    // Hint: Use blockIdx, blockDim, and threadIdx for both x and y
    // Row corresponds to Y axis, Col corresponds to X axis
    // int row = ...
    // int col = ...

    // 2. Perform guard check (make sure row and col are within N)
    
    // 3. Compute dot product of row A and column B
    // float value = 0.0f;
    // for (int k = 0; k < N; k++) {
    //     value += A[row * N + k] * B[k * N + col];
    // }
    // C[row * N + col] = value;
}

int main() {
    // Matrix size N x N
    const int N = 512; 
    const size_t bytes = N * N * sizeof(float);

    // Host memory
    std::vector<float> h_A(N * N, 1.0f); // Fill with 1.0
    std::vector<float> h_B(N * N, 2.0f); // Fill with 2.0
    std::vector<float> h_C(N * N, 0.0f);

    /**
     * STAGE 1: Device Memory Allocation
     * TODO: Allocate d_A, d_B, and d_C on the GPU using cudaMalloc.
     */
    float *d_A, *d_B, *d_C;
    cudaError errorA = cudaMalloc((void**) &d_A, bytes);
    if(errorA != cudaSuccess){
        printf("Cooked, we have error: %s", cudaGetErrorString(errorA));
        return 1;
    }

    cudaError errorB = cudaMalloc((void**) &d_B, bytes);
    if(errorB != cudaSuccess){
        printf("Cooked, we have error: %s", cudaGetErrorString(errorB));
        cudaFree(d_A);
        return 1;
    }

    cudaError errorC = cudaMalloc((void**) &d_C, bytes);
    if(errorC != cudaSuccess){
        printf("Cooked, we have error: %s", cudaGetErrorString(errorC));
        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }
    /**
     * STAGE 2: Copy Data to Device
     * TODO: Copy h_A and h_B to the GPU using cudaMemcpy.
     */
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);


    /**
     * STAGE 3: Kernel Configuration and Launch
     * For 2D data, we use 'dim3' to define 2D blocks and grids.
     */
    int threads_per_dim = 16; // 16x16 = 256 threads per block
    dim3 threadsPerBlock(threads_per_dim, threads_per_dim);
    dim3 blocksPerGrid((N + threads_per_dim - 1) / threads_per_dim, (N + threads_per_dim - 1) / threads_per_dim);

    // TODO: Launch matrixMul here
    // matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /**
     * STAGE 4: Copy Result Back
     * TODO: Copy d_C back to h_C on the host.
     */
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);


    // STAGE 5: Verification
    std::cout << "Verifying..." << std::endl;
    bool success = true;
    for (int i = 0; i < N * N; i++) {
        // Since A is all 1s and B is all 2s, each element in C should be N * 1 * 2
        if (std::fabs(h_C[i] - (N * 2.0f)) > 1e-5) {
            success = false;
            break;
        }
    }
    
    if (success) std::cout << "SUCCESS! Matrix multiplication correct." << std::endl;
    else std::cout << "FAILED! Result is incorrect." << std::endl;

    /**
     * STAGE 6: Cleanup
     * TODO: Free GPU memory using cudaFree.
     */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

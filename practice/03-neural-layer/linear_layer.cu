#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

/**
 * THE NEURAL LAYER KERNEL
 * 
 * This kernel performs three operations in one go (called "Fusion"):
 * 1. Matrix Multiplication: out = input * weight
 * 2. Bias Addition: out = out + bias
 * 3. ReLU Activation: out = max(0, out)
 * 
 * input:  [M x K]
 * weight: [K x N]
 * bias:   [1 x N] (will be added to each row of the result)
 * output: [M x N]
 */
__global__ void linearLayerKernel(const float* input, const float* weight, const float* bias, float* output, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        
        // 1. Dot Product (Matrix Multiplication)
        for (int i = 0; i < K; ++i) {
            sum += input[row * K + i] * weight[i * N + col];
        }

        // 2. Add Bias
        sum += bias[col];

        // 3. ReLU Activation Function
        // if (sum < 0) sum = 0
        output[row * N + col] = fmaxf(0.0f, sum);
    }
}

int main() {
    // Layer dimensions
    // Input: [4 x 3], Weight: [3 x 2], Bias: [1 x 2], Output: [4 x 2]
    int M = 4; // Batch size
    int K = 3; // Input features
    int N = 2; // Output features (Neurons)

    size_t inputSize = M * K * sizeof(float);
    size_t weightSize = K * N * sizeof(float);
    size_t biasSize = N * sizeof(float);
    size_t outputSize = M * N * sizeof(float);

    // Host data (CPU)
    std::vector<float> h_input = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::vector<float> h_weight = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    std::vector<float> h_bias = {0.5, -1.0};
    std::vector<float> h_output(M * N, 0.0f);

    // Device pointers (GPU VRAM)
    float *d_input, *d_weight, *d_bias, *d_output;

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_weight, weightSize);
    cudaMalloc(&d_bias, biasSize);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, h_input.data(), inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), weightSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), biasSize, cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    std::cout << "Launching Linear Layer Kernel..." << std::endl;
    linearLayerKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_weight, d_bias, d_output, M, N, K);

    cudaMemcpy(h_output.data(), d_output, outputSize, cudaMemcpyDeviceToHost);

    // Simple print of results
    std::cout << "Output (After ReLU):" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_output[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);

    return 0;
}

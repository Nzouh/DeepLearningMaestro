#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

/**
 * MNIST Data Format:
 * The files are in a high-endian binary format.
 *
 * Images File:
 * [offset] [type]          [value]          [description]
 * 0000     32 bit integer  0x00000803(2051) magic number
 * 0004     32 bit integer  60000            number of images
 * 0008     32 bit integer  28               number of rows
 * 0012     32 bit integer  28               number of columns
 * 0016     unsigned byte   ??               pixel 0
 * 0017     unsigned byte   ??               pixel 1
 * ...
 */

// Helper to flip the "Endianness" (MNIST is big-endian, most PCs are
// little-endian)

int reverseInt(int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<unsigned char> readMnistImages(std::string path, int &rows,
                                           int &cols, int &num_images) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cout << "Failed to open: " << path << std::endl;
    return {};
  }

  int magic_number = 0;
  num_images = 0;
  rows = 0;
  cols = 0;

  // Read headers
  file.read((char *)&magic_number, 4);
  magic_number = reverseInt(magic_number);

  file.read((char *)&num_images, 4);
  num_images = reverseInt(num_images);

  file.read((char *)&rows, 4);
  rows = reverseInt(rows);

  file.read((char *)&cols, 4);
  cols = reverseInt(cols);

  std::cout << "Magic Number: " << magic_number << std::endl;
  std::cout << "Images Found: " << num_images << std::endl;
  std::cout << "Resolution: " << rows << "x" << cols << std::endl;

  // TODO: Create a vector to hold ALL pixel data
  // Each pixel is an unsigned char (0-255).
  // The total size is: number_of_images * rows * cols
  size_t total_size = num_images * rows * cols;
  std::vector<unsigned char> pixels(total_size);
  file.read((char *)pixels.data(), total_size);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      unsigned char pixel = pixels[cols * r + c];

      if (pixel > 128)
        std::cout << "#";
      else
        std::cout << ".";
    }
    std::cout << std::endl;
  }
  return pixels;
}

std::vector<unsigned char> readMnistLabels(std::string path, int &count) {
  std::ifstream file(path, std::ios::binary);

  if (!file.is_open()) {
    std::cout << "Failed to open: " << path << std::endl;
    return {};
  }

  int magic_number = 0;
  count = 0;
  file.read((char *)&magic_number, 4);
  file.read((char *)&count, 4);
  magic_number = reverseInt(magic_number);
  count = reverseInt(count);

  std::vector<unsigned char> labels(count);
  file.read((char *)labels.data(), count);

  std::cout << "Magic Number (Labels): " << magic_number << std::endl;
  std::cout << "Labels Found: " << count << std::endl;
  std::cout << "Label for image 0: " << (int)labels[0] << std::endl;

  return labels;
}

// TASK:
// 1. Allocate a std::vector<unsigned char> pixels
// 2. file.read((char*)pixels.data(), total_size)
// 3. Print out the pixel values of the first image to see if it looks like
// data

/**
 * Normalizer and Preparation
 * This function takes the raw binary bytes and converts them to
 * the floating point format the GPU expects.
 *
 * NOTE: We use std::vector<float> because the GPU math is optimized
 * for 32-bit floating point numbers.
 */
std::vector<float>
normalizePixels(const std::vector<unsigned char> &raw_pixels) {
  std::vector<float> normalized(raw_pixels.size());
  size_t N = raw_pixels.size();
  // Find the mean
  double sum = 0;
  for (unsigned char p : raw_pixels) {
    sum += p;
  }

  float mean = float(sum / N);

  // Calculate Standard Deviation
  double sum_of_squares = 0.0;
  for (unsigned char p : raw_pixels) {
    sum_of_squares += std::pow(p - mean, 2);
  }

  double standard_deviation = std::pow(sum_of_squares / (N - 1), 0.5);

  for (size_t i = 0; i < N; i++) {
    normalized[i] = (float)((raw_pixels[i] - mean) / standard_deviation);
  }

  return normalized;
}

__device__ float relu(float x) { return x > 0 ? x : 0; }

__global__ void computationKernel(const float *weights, const float *biases,
                                  float *inputs, float *outputs,
                                  int num_neurons, int num_inputs,
                                  bool apply_relu) {
  int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (neuron_idx < num_neurons) {
    float sum = 0.0f;
    for (int i = 0; i < num_inputs; i++) {
      sum += weights[neuron_idx * num_inputs + i] * inputs[i];
    }
    float out_val = sum + biases[neuron_idx];
    outputs[neuron_idx] = apply_relu ? relu(out_val) : out_val;
  }
}

__global__ void softmaxKernel(float *outputs, int num_outputs) {
  // Single-threaded softmax for simplicity on 10 outputs
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float max_val = outputs[0];
    for (int i = 1; i < num_outputs; i++) {
      if (outputs[i] > max_val)
        max_val = outputs[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < num_outputs; i++) {
      outputs[i] =
          expf(outputs[i] - max_val); // subtract max for numerical stability
      sum_exp += outputs[i];
    }

    for (int i = 0; i < num_outputs; i++) {
      outputs[i] /= sum_exp;
    }
  }
}

__global__ void backpropOutputLayer(unsigned char *labels, float *weights2,
                                    float *biases2, float *hidden_outputs,
                                    float *final_outputs,
                                    int num_outputs, int num_hidden,
                                    int image_index) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_outputs) {
    float target = (labels[image_index] == (unsigned char)idx) ? 1.0f : 0.0f;

    // Softmax + Cross Entropy derivative beautifully simplifies to:
    // (predicted_probability - true_probability)
    float dz = final_outputs[idx] - target;

    float learning_rate =
        0.001f; // Raised back up now that Softmax prevents exploding gradients
    biases2[idx] -= learning_rate * dz;
    for (int i = 0; i < num_hidden; i++) {
      weights2[idx * num_hidden + i] -= learning_rate * dz * hidden_outputs[i];
    }
  }
}

// Computes error for Hidden Layer based on Output errors, and updates Layer 1
// weights/biases
__global__ void backpropHiddenLayer(float *weights1, float *biases1,
                                    float *weights2, float *inputs,
                                    float *hidden_outputs, float *final_outputs,
                                    unsigned char *labels,
                                    int num_hidden, int num_outputs,
                                    int num_inputs, int image_index) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_hidden) {
    // Calculate how much this hidden neuron contributed to the output errors
    float error = 0.0f;
    for (int j = 0; j < num_outputs; j++) {
      float target = (labels[image_index] == (unsigned char)j) ? 1.0f : 0.0f;
      float dz_out = final_outputs[j] - target;
      error += weights2[j * num_hidden + idx] * dz_out;
    }

    float dz = (hidden_outputs[idx] > 0.0f) ? error : 0.0f; // ReLU derivative

    float learning_rate =
        0.001f; // Safe to increase because output layer is using Softmax
    if (dz != 0.0f) {
      biases1[idx] -= learning_rate * dz;
      for (int i = 0; i < num_inputs; i++) {
        weights1[idx * num_inputs + i] -= learning_rate * dz * inputs[i];
      }
    }
  }
}

int main() {
  int rows = 0, cols = 0, num_images = 0, count = 0;
  std::vector<unsigned char> raw_pixels =
      readMnistImages("train-images.idx3-ubyte", rows, cols, num_images);
  std::vector<unsigned char> labels =
      readMnistLabels("train-labels.idx1-ubyte", count);
  // 1. Logic to Load Raw Files (Copy functions from mnist_loader.cpp here
  // later)
  // 2. Normalize
  std::vector<float> pixels = normalizePixels(raw_pixels);

  int num_pixels = rows * cols; // 784 for MNIST

  // --- NEW: SHUFFLE DATA ---
  std::cout << "Shuffling data..." << std::endl;
  std::vector<int> indices(count);
  for (int i = 0; i < count; i++) indices[i] = i;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(indices.begin(), indices.end(), gen);

  std::vector<float> s_pixels(pixels.size());
  std::vector<unsigned char> s_labels(labels.size());
  for (int i = 0; i < count; i++) {
    for (int j = 0; j < num_pixels; j++) {
      s_pixels[i * num_pixels + j] = pixels[indices[i] * num_pixels + j];
    }
    s_labels[i] = labels[indices[i]];
  }
  pixels = s_pixels;
  labels = s_labels;

  int num_neurons = 128;        // Assuming you want 128 hidden neurons

  // --- ELEMENT COUNTS (Use these for vectors and loops) ---
  size_t weight_count = (size_t)num_pixels * num_neurons;
  size_t bias_count = (size_t)num_neurons;
  size_t input_count = pixels.size();

  // --- BYTE SIZES (Use these ONLY for cudaMalloc and cudaMemcpy) ---
  size_t weight_bytes = weight_count * sizeof(float);
  size_t bias_bytes = bias_count * sizeof(float);
  size_t input_bytes = input_count * sizeof(float);

  size_t label_bytes = (size_t)count * sizeof(unsigned char);
  size_t output_bytes = (size_t)num_neurons * sizeof(float);

  // d_A is a pointer for inputs
  // d_B is a pointer for weights
  // d_C is a pointer for bias
  float *d_A, *d_B, *d_B2, *d_C, *d_C2, *d_Outputs, *d_FinalOutputs;
  unsigned char *d_Labels;

  // Creating random weights and biases (these are set to 0)
  std::vector<float> biases(bias_count, 0.0f);
  std::vector<float> weights(weight_count);
  float std_dev = std::sqrt(2.0f / num_pixels);
  std::normal_distribution<float> dis_weight(0, std_dev);

  for (auto &w : weights)
    w = dis_weight(gen);

  // Allocate the d_A pointer to the GPU, which will allocate the number of
  // bytes
  cudaError errorA = cudaMalloc((void **)&d_A, input_bytes);
  if (errorA != cudaSuccess) {
    std::cout << "d_A could not be properly loaded"
              << cudaGetErrorString(errorA);
  }

  // Allocation for Layer 1
  cudaError errorB = cudaMalloc((void **)&d_B, weight_bytes);
  if (errorB != cudaSuccess) {
    std::cout << "d_B could not be properly loaded"
              << cudaGetErrorString(errorB);
    cudaFree(d_A);
  }

  cudaError errorC = cudaMalloc((void **)&d_C, bias_bytes);
  if (errorC != cudaSuccess) {
    std::cout << "d_C could not be properly loaded"
              << cudaGetErrorString(errorC);
    cudaFree(d_A);
    cudaFree(d_B);
  }

  // --- LAYER 2: HIDDEN (128) -> OUTPUT (10) ---
  int num_outputs = 10;
  size_t weight_count2 = (size_t)num_neurons * num_outputs;
  size_t weight_bytes2 = weight_count2 * sizeof(float);
  size_t bias_bytes2 = num_outputs * sizeof(float);
  size_t final_output_bytes = num_outputs * sizeof(float);

  cudaMalloc((void **)&d_B2, weight_bytes2);
  cudaMalloc((void **)&d_C2, bias_bytes2);
  cudaMalloc((void **)&d_FinalOutputs, final_output_bytes);

  // Initialize Layer 2 Weights (He Initialization)
  std::vector<float> weights2(weight_count2);
  std::vector<float> biases2(num_outputs, 0.0f);
  float std_dev2 = std::sqrt(2.0f / num_neurons);
  std::normal_distribution<float> dis_weight2(0, std_dev2);
  for (auto &w : weights2)
    w = dis_weight2(gen);

  cudaMemcpy(d_B2, weights2.data(), weight_bytes2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C2, biases2.data(), bias_bytes2, cudaMemcpyHostToDevice);

  cudaError errorD = cudaMalloc((void **)&d_Labels, label_bytes);
  if (errorD != cudaSuccess) {
    std::cout << "d_Labels could not be properly loaded"
              << cudaGetErrorString(errorD);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  }

  cudaError errorOutputs = cudaMalloc((void **)&d_Outputs, output_bytes);
  if (errorOutputs != cudaSuccess) {
    std::cout << "d_Outputs could not be properly loaded"
              << cudaGetErrorString(errorOutputs);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_Labels);
  }

  cudaMemcpy(d_A, pixels.data(), input_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, weights.data(), weight_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, biases.data(), bias_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Labels, labels.data(), label_bytes, cudaMemcpyHostToDevice);

  int threadsPerBlock = 128;
  int blocks1 = (num_neurons + threadsPerBlock - 1) / threadsPerBlock;
  int blocks2 = (num_outputs + threadsPerBlock - 1) / threadsPerBlock;

  // Run through the entire dataset 10 times (10 Epochs)
  int epochs = 10;
  int train_count = count - 10000; // Leave 10,000 for validation!
  for (int epoch = 0; epoch < epochs; epoch++) {
    for (int img = 0; img < train_count; img++) {
      // Offset the d_A pointer to point to the start of the current image
      float *current_image = &d_A[img * num_pixels];

      // 1. Forward Pass: Layer 1 (Input -> Hidden) applies ReLU
      computationKernel<<<blocks1, threadsPerBlock>>>(
          d_B, d_C, current_image, d_Outputs, num_neurons, num_pixels, true);

      // 2. Forward Pass: Layer 2 (Hidden -> Output) is strictly linear
      computationKernel<<<blocks2, threadsPerBlock>>>(
          d_B2, d_C2, d_Outputs, d_FinalOutputs, num_outputs, num_neurons,
          false);

      // 2.5 Softmax Activation
      softmaxKernel<<<1, 1>>>(d_FinalOutputs, num_outputs);

      // 3. Backward Pass
      // Step A: Update Hidden Layer weights (using probabilities BEFORE weights2 are changed)
      backpropHiddenLayer<<<blocks1, threadsPerBlock>>>(
          d_B, d_C, d_B2, current_image, d_Outputs, d_FinalOutputs, d_Labels,
          num_neurons, num_outputs, num_pixels, img);

      // Step B: Update Output Layer weights
      backpropOutputLayer<<<blocks2, threadsPerBlock>>>(
          d_Labels, d_B2, d_C2, d_Outputs, d_FinalOutputs,
          num_outputs, num_neurons, img);

      // --- EVALUATION CHECK (Every 10,000 images) ---
      if (img % 10000 == 0) {
        cudaDeviceSynchronize();
        std::vector<float> h_outputs(num_outputs);
        cudaMemcpy(h_outputs.data(), d_FinalOutputs,
                   num_outputs * sizeof(float), cudaMemcpyDeviceToHost);

        int predicted_digit = 0;
        float max_confidence = h_outputs[0];

        for (int i = 1; i < num_outputs; i++) {
          if (h_outputs[i] > max_confidence) {
            max_confidence = h_outputs[i];
            predicted_digit = i;
          }
        }

        std::cout << "Epoch " << epoch << " | Img " << img
                  << " | Target: " << (int)labels[img]
                  << " | Predicted: " << predicted_digit
                  << " (Confidence: " << max_confidence << ")" << std::endl;
      }
    }
  }

  cudaDeviceSynchronize();
  std::vector<float> h_outputs(num_outputs);
  cudaMemcpy(h_outputs.data(), d_FinalOutputs, num_outputs * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "Successfully ran the training kernel loops." << std::endl;

  // ==========================================
  // --- FINAL TESTING / EVALUATION PHASE -----
  // ==========================================
  std::cout << "\n=============================================" << std::endl;
  std::cout << "Training Complete! Testing on 10,000 holdout images..."
            << std::endl;

  int test_count = 10000;
  int correct_predictions = 0;

  for (int img = train_count; img < count; img++) {
    float *current_image = &d_A[img * num_pixels];

    // 1. Forward Pass Only! (NO BACKPROPAGATION)
    computationKernel<<<blocks1, threadsPerBlock>>>(
        d_B, d_C, current_image, d_Outputs, num_neurons, num_pixels, true);
    computationKernel<<<blocks2, threadsPerBlock>>>(
        d_B2, d_C2, d_Outputs, d_FinalOutputs, num_outputs, num_neurons, false);
    softmaxKernel<<<1, 1>>>(d_FinalOutputs, num_outputs);

    // 2. Read Results
    cudaDeviceSynchronize();
    cudaMemcpy(h_outputs.data(), d_FinalOutputs, num_outputs * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 3. Find Prediction
    int predicted_digit = 0;
    float max_val = h_outputs[0];
    for (int i = 1; i < num_outputs; i++) {
      if (h_outputs[i] > max_val) {
        max_val = h_outputs[i];
        predicted_digit = i;
      }
    }

    // 4. Check if correct
    if (predicted_digit == (int)labels[img])
      correct_predictions++;
  }

  float accuracy = (correct_predictions / (float)test_count) * 100.0f;
  std::cout << "\n>>> FINAL TEST ACCURACY: " << accuracy << "% ("
            << correct_predictions << "/" << test_count << " correct) <<<"
            << std::endl;
  std::cout << "=============================================\n" << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_Outputs);
  cudaFree(d_Labels);
  cudaFree(d_B2);
  cudaFree(d_C2);
  cudaFree(d_FinalOutputs);

  return 0;
}

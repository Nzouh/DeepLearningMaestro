#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>


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

std::vector<unsigned char> readMnistImages(std::string path, int &rows, int &cols, int &num_images) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cout << "Failed to open: " << path << std::endl;
    return {};
  }

  int magic_number = 0;
  num_images= 0;
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
    
    if(!file.is_open()){
        std::cout << "Failed to open: " << path << std::endl;
        return {};
    }

    int magic_number = 0;
    count = 0;
    file.read((char*) &magic_number, 4);
    file.read((char*) &count, 4);
    magic_number = reverseInt(magic_number);
    count = reverseInt(count);
    std::vector<unsigned char> labels(count);
    file.read((char*)labels.data(), count);

    std::cout << "Magic Number (Labels): " << magic_number << std::endl;
    std::cout << "Labels Found: " << count << std::endl;
    std::cout << "Label for image 0: " << (int)labels[0] << std::endl;

    return labels;
    // 1. Open file
    // 2. Read Magic Number (2049) and Count (60,000)
    // 3. reverseInt() both of them
    // 4. Create std::vector<unsigned char> labels(count)
    // 5. file.read((char*)labels.data(), count)
    // 6. std::cout << "Label for image 0: " << (int)labels[0] << std::endl;
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
std::vector<float> normalizePixels(const std::vector<unsigned char>& raw_pixels) {
    std::vector<float> normalized(raw_pixels.size());
    size_t N = raw_pixels.size();
    //Find the mean
    double sum = 0;
    for(unsigned char p : raw_pixels){
        sum += p;
    }
    
    float mean = float(sum / N);

    //Calculate Standard Deviation
    double sum_of_squares= 0.0;
    for(unsigned char p: raw_pixels){
        sum_of_squares += std::pow(p - mean, 2);
    }

    double standard_deviation = std::pow(sum_of_squares / (N-1), 0.5);

    for(size_t i = 0; i < N; i ++){
        normalized[i] = ((raw_pixels[i] - mean) / standard_deviation);
    }

    
    return normalized;
}

int main() {
    int rows = 0, cols = 0, num_images = 0, count = 0;
    std::vector<unsigned char> raw_pixels = readMnistImages("train-images.idx3-ubyte", rows, cols, num_images);
    std::vector<unsigned char> labels = readMnistLabels("train-labels.idx-1-ubyte", count);
    // 1. Logic to Load Raw Files (Copy functions from mnist_loader.cpp here later)
    // 2. Normalize
    std::vector<float> pixels = normalizePixels(raw_pixels);

    int num_pixels = rows * cols; // 784 for MNIST
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


    

    // 3. Move to GPU (cudaMalloc/cudaMemcpy)
    //d_A is a pointer for inputs
    //d_B is a pointer for weights
    //d_C is a pointer for bias
    float *d_A, *d_B, *d_C;
    unsigned char *d_D;

    //Creating random weights and biases (these are set to 0)
    std::vector<float> biases(bias_count, 0.0f);
    std::vector<float> weights(weight_count);
    float std_dev = std::sqrt(2.0f / num_pixels);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis_weight(0, std_dev);

    for(auto& w : weights) w = dis_weight(gen);


    //Allocate the d_A pointer to the GPU, which will allocate the number of bytes
    cudaError errorA = cudaMalloc((void**) &d_A, input_bytes);
    if(errorA != cudaSuccess){
        std::cout << "d_A could not be properly loaded" << cudaGetErrorString(errorA);
    }

    cudaError errorB = cudaMalloc((void**) &d_B, weight_bytes);
    if(errorB != cudaSuccess){
        std::cout << "d_B could not be properly loaded" << cudaGetErrorString(errorB);
        cudaFree(d_A);
    }

    cudaError errorC = cudaMalloc((void**) &d_C, bias_bytes);
    if(errorC != cudaSuccess){
        std::cout << "d_C could not be properly loaded" << cudaGetErrorString(errorC);
        cudaFree(d_A);
        cudaFree(d_B);
    }

    cudaError errorD = cudaMalloc((void**) &d_D, label_bytes);
    if(errorD != cudaSuccess){
        std::cout << "d_D could not be properly loaded" << cudaGetErrorString(errorD);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }


    cudaMemcpy(d_A, pixels.data(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, weights.data(), weight_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, biases.data(), bias_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, labels.data(), label_bytes, cudaMemcpyHostToDevice);

    return 0;
}


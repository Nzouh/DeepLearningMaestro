#include <fstream>
#include <iostream>
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

void readMnistImages(std::string path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cout << "Failed to open: " << path << std::endl;
    return;
  }

  int magic_number = 0;
  int number_of_images = 0;
  int rows = 0;
  int cols = 0;

  // Read headers
  file.read((char *)&magic_number, 4);
  magic_number = reverseInt(magic_number);

  file.read((char *)&number_of_images, 4);
  number_of_images = reverseInt(number_of_images);

  file.read((char *)&rows, 4);
  rows = reverseInt(rows);

  file.read((char *)&cols, 4);
  cols = reverseInt(cols);

  std::cout << "Magic Number: " << magic_number << std::endl;
  std::cout << "Images Found: " << number_of_images << std::endl;
  std::cout << "Resolution: " << rows << "x" << cols << std::endl;

  // TODO: Create a vector to hold ALL pixel data
  // Each pixel is an unsigned char (0-255).
  // The total size is: number_of_images * rows * cols
  size_t total_size = number_of_images * rows * cols;
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
}

void readMnistLabels(std::string path) {
    std::ifstream file(path, std::ios::binary);
    
    if(!file.is_open()){
        std::cout << "Failed to open: " << path << std::endl;
        return;
    }

    int magic_number=0;
    int count = 0;
    file.read((char*) &magic_number, 4);
    file.read((char*) &count, 4);
    magic_number = reverseInt(magic_number);
    count = reverseInt(count);
    std::vector<unsigned char> labels(count);
    file.read((char*)labels.data(), count);

    std::cout << "Magic Number (Labels): " << magic_number << std::endl;
    std::cout << "Labels Found: " << count << std::endl;
    std::cout << "Label for image 0: " << (int)labels[0] << std::endl;

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


int main() {
  // You will need to download 'train-images-idx3-ubyte' from MNIST website
  // and place it in this folder.
  readMnistImages("train-images.idx3-ubyte");
  readMnistLabels("train-labels.idx1-ubyte");
  return 0;
}

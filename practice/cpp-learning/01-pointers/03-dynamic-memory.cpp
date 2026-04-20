#include <iostream>

/**
 * LESSON 3: Dynamic Memory (The Heap)
 * 
 * In Lesson 2, we used 'int numbers[5]'. This is "Stack" memory.
 * It has a fixed size and disappears when the function ends.
 * 
 * In Deep Learning (and real software), you often don't know the size 
 * until the program runs (e.g., number of pixels in an image).
 * 
 * For this, we use the "Heap" via:
 * 1. 'new'    -> Asks the OS for a block of memory.
 * 2. 'delete' -> Returns that memory to the OS.
 * 
 * CRITICAL RULE: Every 'new' must have a 'delete'. Failure to delete 
 * results in a "Memory Leak" (your program eats RAM until it crashes).
 */

int main(int argc, char** argv) {
    int n;
    std::cout << "Enter the size of the array: ";
    // std::cin is the conveyor belt for INPUT (Opposite of cout)
    if (!(std::cin >> n)) {
        std::cout << "Invalid input!" << std::endl;
        return 1;
    }

    // 1. ALLOCATION
    // We create a pointer to hold the address of the new block.
    // 'new int[n]' asks for 'n' integers in a row.
    int* my_dynamic_array = new int[n];

    std::cout << "Memory allocated at: " << my_dynamic_array << std::endl;

    // 2. USAGE
    // You can use the pointer just like an array!
    for (int i = 0; i < n; i++) {
        my_dynamic_array[i] = i * 10;
        std::cout << "Element " << i << ": " << my_dynamic_array[i] << std::endl;
    }

    // ---------------------------------------------------------
    // CHALLENGE: Manual Memory Management
    // ---------------------------------------------------------
    
    // TODO 1: Create a second dynamic array of floats named 'gpu_mimic'.
    //         Size should be 'n'.
    
    float*  gpu_mimic = new float[n];
    std::cout << "Memory allocated at: " << gpu_mimic << std::endl;

    // TODO 2: Fill it with values (e.g. i / 2.0).

    for(int i = 0; i < n; i ++){
        gpu_mimic[i] = i / 2.0f;
    }

    // TODO 3: Print the first and last element of 'gpu_mimic'.

    printf("First element: %f\nLast element: %f\n",gpu_mimic[0], gpu_mimic[n-1]);

    /**
     * TODO 4: DEALLOCATION (Very Important)
     * You must free the memory for BOTH arrays.
     * Use 'delete[] my_dynamic_array;' (The [] is needed for arrays).
     * Then do the same for 'gpu_mimic'.
     */


    std::cout << "Cleaning up memory..." << std::endl;

    delete[] my_dynamic_array;
    delete[] gpu_mimic;

    // Write your delete calls here...

    return 0;
}

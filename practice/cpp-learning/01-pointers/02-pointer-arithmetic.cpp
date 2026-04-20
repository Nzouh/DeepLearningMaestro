#include <iostream>

/**
 * LESSON 2: Pointers and Arrays
 * 
 * In C++, the name of an array is actually just a pointer 
 * to its FIRST element.
 * 
 * Key Concept: Pointer Arithmetic
 * If ptr points to array[0], then (ptr + 1) points to array[1].
 * The computer knows to jump the correct number of bytes 
 * based on the TYPE (e.g. 4 bytes for an int).
 */

int main(int argc, char** argv) {
    // 1. Create an array
    int numbers[5] = {10, 20, 30, 40, 50};
    
    // 2. Point to the start of the array
    int* ptr = numbers; // Note: No '&' needed for arrays!

    std::cout << "--- Pointer Arithmetic ---" << std::endl;
    
    // 3. Accessing via pointer arithmetic
    std::cout << "First element (ptr):      " << *ptr << std::endl;
    std::cout << "Second element (ptr + 1):  " << *(ptr + 1) << std::endl;
    std::cout << "Third element (ptr + 2):   " << *(ptr + 2) << std::endl;

    // 4. Changing values via pointer
    *(ptr + 4) = 500; // Changes the 5th element
    std::cout << "Modified 5th element:     " << numbers[4] << std::endl;

    std::cout << "--------------------------" << std::endl;

    /**
     * CHALLENGE: The Reverse Loop
     * 
     * TODO 1: Create a pointer named 'end_ptr' that points to the LAST element (numbers[4]).
     * 
     * TODO 2: Use a while loop and that pointer to print the array BACKWARDS.
     *         Inside the loop, you should:
     *         - Print the value (*end_ptr)
     *         - Move the pointer backwards (end_ptr--)
     * TODO 3: The loop should stop when 'end_ptr' is less than the 'numbers' start address.
     */
    std::cout << "Numbers in Reverse\n";
    int* end_ptr = &numbers[4];
    while (end_ptr >= ptr){
        printf("%i\n", *end_ptr);
        end_ptr--;
    }

    return 0;
}

#include <iostream>

/**
 * LESSON 1: Streams and Pointers
 * 
 * Part A: Streams (std::cout <<)
 * In Python, you have print("Hello").
 * In C++, you have a "stream". Think of 'std::cout' as the "Screen".
 * The '<<' operator is the "Inlet" pipe. 
 * 'std::endl' is the "NewLine" pipe.
 * 
 * Part B: Pointers (* and &)
 * A pointer is just an integer that holds a memory address.
 * '&' = "Get the address of" (The physical location in RAM)
 * '*' = "Value at this address" (Follow the link to the treasure)
 */

int main(int argc, char** argv) {
    // 1. Standard variables (stored in a box)
    int coffee_price = 5;
    
    // 2. Pointers (stored as an address to that box)
    // The '*' here means "this variable is a pointer type"
    int* ptr_to_coffee = &coffee_price; // '&' gets the address of the box

    // 3. Printing with Streams (std::cout)
    // You "push" things into the stream one by one.
    std::cout << "--- COFFEE STATS ---\n";
    std::cout << "Direct Value of coffee_price: " << coffee_price << "\n";
    
    // This will print a hexadecimal address (e.g., 0x7ffe...)
    std::cout << "Address of coffee_price (&):  " << &coffee_price << "\n";
    
    // This prints the same address because ptr_to_coffee HOLDS it
    std::cout << "Value of ptr_to_coffee:       " << ptr_to_coffee << "\n";

    // 4. Dereferencing (The magic of *)
    // When we use '*' on a variable that ALREADY exists, it means "go to that address"
    std::cout << "Value AT that address (*ptr): " << *ptr_to_coffee << "\n";

    std::cout << "--------------------" << "\n";

    // TODO 1: Create an integer variable named 'my_age'.
    int my_age = 21;
    std::cout << "my age is " << my_age << "\n";
    // TODO 2: Create a pointer named 'age_ptr' that points to 'my_age'.
    int* age_ptr = &my_age;
    // TODO 3: Print 'my_age' using the pointer.
    std::cout << "printing my age using the pointer: " << *age_ptr << "\n";
    // TODO 4: Change 'my_age' by using the pointer (e.g. *age_ptr = 25;) 
    *age_ptr = 23;
    std::cout << "Changing my age using the pointer: " <<*age_ptr << "\n";
    //         and print 'my_age' again.
    
    return 0;
}

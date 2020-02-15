#include <iostream>

/**
 * This is a demo to show you how to apply homomorphic encryption to machine learning.
 * We implement the algorithm named KNN to recognize hand written numbers between 0 to 9.
 * This program uses the MNIST data to train and test.
 * The homomorphic encryption is based on the library named SEAL which has been opened source by Microsoft.
 * 
 **/

int main()
{
    uint32_t num;
    while (true)
    {
        std::cout << "please select:" << std::endl;
        std::cout << "1. train:" << std::endl;
        std::cout << "2. test:" << std::endl;
        std::cout << "3. recognize:" << std::endl;
        std::cin >> num >> std::endl;
        switch (num)
        {
        case 1:
        case 2:
        case 3:
        default:
        {
            std::cout << "please select 1,2 or 3. Try again!"
        }
        }
    }
    return 0;
}
#include <iostream>
#include <string>

/**
 * This is a demo to show you how to apply homomorphic encryption to machine learning.
 * We implement the algorithm named KNN to recognize hand written numbers between 0 to 9.
 * This program uses the MNIST data to train and test.
 * The homomorphic encryption is based on the library named SEAL which has been opened source by Microsoft.
 * 
 **/
#include "knn.h"

int main()
{
    uint32_t num;
    while (true)
    {
        cout << "please select:" << endl;
        cout << "1. test" << endl;
        cout << "2. recognize" << endl;
        cout << "3. exit" << endl;
        cin >> num;
        switch (num)
        {
        case 1:
        {
            return 0;
        }
        case 2:
        {
            return 0;
        }
        case 3:
        {
            return 0;
        }
        default:
        {
            cout << "please select 1, 2 or 3. Try again!" << endl;
        }
        }
    }
    return 0;
}
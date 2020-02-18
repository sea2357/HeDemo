#include <iostream>
#include <string>
#include "knn.h"
/**
 * This is a demo to show you how to apply homomorphic encryption to machine learning.
 * We implement the algorithm named KNN to recognize hand written numbers between 0 to 9.
 * This program uses the MNIST data to train and test.
 * The homomorphic encryption is based on the library named SEAL which has been opened source by Microsoft.
 * 
 **/
using namespace std;
using namespace cv;

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
            KNN knn;
            string data_path = "../data";
            knn.test(data_path);
            return 0;
        }
        case 2:
        {
            KNN knn;
            std::string pic;
            string data_path = "../data";
            cout << "please specify a image file " << endl;
            cin >> pic;
            knn.recognize(data_path, pic);
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
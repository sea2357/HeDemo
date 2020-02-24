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
        KNN knn;
        string pic;
        string data_path = "../data";
        size_t num = 70;
        cout << "please select:" << endl;
        cout << "1. test using default parameters" << endl;
        cout << "2. recognize a plaintext picture" << endl;
        cout << "3. recognize an encrypted picture" << endl;
        cout << "4. test the success rate" << endl;
        cout << "5. exit" << endl;
        cin >> num;
        switch (num)
        {
        case 1:
        {
            pic = "../data/4.bmp";
            cout << "recognize 4.bmp using plaintext......" << endl;
            knn.recognize(data_path, pic, num);
            cout << "recognize 4.bmp using ciphertext......" << endl;
            knn.ciphertext_recognize(data_path, pic, num);
            return 0;
        }
        case 2:
        {
            cout << "please specify a image file, e.g. ../data/4.bmp " << endl;
            cin >> pic;
            size_t num = 70;
            knn.recognize(data_path, pic, num);
            return 0;
        }
        case 3:
        {
            cout << "please specify a image file, e.g. ../data/4.bmp " << endl;
            cin >> pic;
            size_t num = 70;
            knn.ciphertext_recognize(data_path, pic, num);
            return 0;
        }
        case 4:
        {
            knn.test(data_path);
        }
        case 5:
        {
            return 0;
        }
        default:
        {
            cout << "please select 1, 2, 3, 4 or 5. Try again!" << endl;
        }
        }
    }
    return 0;
}
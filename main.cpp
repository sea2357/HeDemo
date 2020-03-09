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
    uint32_t s;
    KNN knn;
    string pic;
    string data_path;
    size_t num = 70;
    while (true)
    {
        cout << "1. recognize a plaintext picture" << endl;
        cout << "2. recognize an encrypted picture" << endl;
        cout << "3. recognize an encrypted picture, and compress 4096 train images into one plaintext" << endl;
        cout << "4. test the success rate" << endl;
        cout << "5. exit" << endl;
        cout << "please select: ";
        cin >> s;
        switch (s)
        {
        case 1:
        {
            cout << "please input the path of train images, e.g. ../data " << endl;
            cin >> data_path;
            cout << "please specify an image file  to recognize, e.g. ../data/4.bmp " << endl;
            cin >> pic;
            size_t num = 4096;
            knn.recognize(data_path, pic, num);
            return 0;
        }
        case 2:
        {
            cout << "please input the path of train images, e.g. ../data " << endl;
            cin >> data_path;
            cout << "please specify an image file  to recognize, e.g. ../data/4.bmp " << endl;
            cin >> pic;
            size_t num = 70;
            knn.ciphertext_recognize(data_path, pic, num);
            return 0;
        }
        case 3:
        {
            cout << "please input the path of train images, e.g. ../data " << endl;
            cin >> data_path;
            cout << "please specify an image file to recognize, e.g. ../data/4.bmp " << endl;
            cin >> pic;
            knn.ciphertext_recognize_compressed(data_path, pic);
            return 0;
        }
        case 4:
        {
            cout << "Please input the number  of train images to be used. Remember that the more images used, the higher of the success rate." << endl;
            while (true)
            {
                cout << "num = ";
                cin >> num;
                if ((num > 10000) || (num <= 0))
                {
                    std::cout << "please input num > 0 and num < 10000 " << std::endl;
                }
                else
                {
                    break;
                }
            }
            knn.test(data_path, num);
            return 0;
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
}
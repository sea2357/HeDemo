
#pragma once
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "seal.h"

/**
 * @brief Implement the algorithm named K-NearestNeighbor.
 * 
 */
class KNN
{

private:
    /**
     * @brief reverse the integer.
     * 
     * @param i : input the integer.
     * @return int : ouput an integer which is reversed.
     */
    int
    reverseInt(int i);
    /**
     * @brief read the image of mnist data.
     * 
     * @param fileName : input the file name of mnist data.
     * @param num [in] the number of train pictures to be used.
     * @return Mat : matrix data of the images in mnist data.
     */
    cv::Mat read_mnist_image(const std::string fileName, const size_t num = 70);
    /**
     * @brief read the label of mnist data.
     * 
     * @param fileName : input the file name of mnist data.
     * @param num [in] the number of train pictures to be used.
     * @return Mat : matrix data of the labels in mnist data.
     */
    //
    cv::Mat read_mnist_label(const std::string fileName, const size_t num = 70);

public:
    KNN(/* args */);

    ~KNN();
    /**
     * @brief Core part of KNN.
     * 
     * @param train_labels [in] labels of train data in mnist.
     * @param train_images [in] images of train data in mnist.
     * @param test_image   [in] the image to be recognized.
     * @param num [in] the number of train pictures to be used.
     * @return std::vector<std::pair<float, unsigned int>> [out] scores correspond to train_labels.
     */
    std::vector<std::pair<float, unsigned int>>
    core(const cv::Mat &train_labels, const cv::Mat &train_images, const cv::Mat &test_image, const size_t num = 70);

    /**
     * @brief Test the success rate of recognzie handwritten number which is between 0 and 9 using KNN algorithm.
     * 
     * @param data_path [in] the path of mnist data.
     * @param num [in] the number of train pictures to be used.
     */
    void test(const std::string &data_path, const size_t num);
    /**
     * @brief Recognize a picture contains a handwritten number which is between 0 and 9.
     * @param data_path [in] the path of mnist data.
     * @param filename [in] the name of the picture to be recognize.
     * @param num [in] the number of train pictures to be used.
     * @return int [out] 0 is success, otherwise is fail.
     */
    int recognize(const std::string &data_path, const std::string &filename, const size_t num = 70);

    /**
     * @brief Recognize a encrypted picture contains a handwritten number which is between 0 and 9.
     * @param data_path [in] the path of mnist data.
     * @param filename [in] the name of the picture to be recognize.
     * @param num [in] the number of train pictures to be used.
     * @return int [out] 0 is success, otherwise is fail.
     */
    int ciphertext_recognize(const std::string &data_path, const std::string &filename, const size_t num = 70);

    /**
     * @brief Recognize a encrypted picture contains a handwritten number which is between 0 and 9. Moreover each bits from 4096 train images is packaged on the same ciphertext.
     * @param data_path [in] the path of mnist data.
     * @param filename [in] the name of the picture to be recognize.
     * @param num [in] the number of train pictures to be used.
     * @return int [out] 0 is success, otherwise is fail.
     */
    int ciphertext_recognize_compressed(const std::string &data_path, const std::string &filename);
};

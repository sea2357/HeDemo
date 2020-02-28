#include "knn.h"

using namespace std;
using namespace cv;
using namespace seal;

KNN::KNN()
{
}

KNN::~KNN()
{
}

//计时器
double cost_time;
clock_t start_time;
clock_t end_time;

int KNN::reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

Mat KNN::read_mnist_image(const string fileName, const size_t num)
{
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    Mat DataMat;

    ifstream file(fileName, ios::binary);
    if (file.is_open())
    {
        cout << "成功打开图像集 ... \n";

        file.read((char *)&magic_number, sizeof(magic_number));
        file.read((char *)&number_of_images, sizeof(number_of_images));
        file.read((char *)&n_rows, sizeof(n_rows));
        file.read((char *)&n_cols, sizeof(n_cols));

        magic_number = reverseInt(magic_number);
        number_of_images = num;
        n_rows = reverseInt(n_rows);
        n_cols = reverseInt(n_cols);

        start_time = clock();
        DataMat = Mat::zeros(number_of_images, n_rows * n_cols, CV_32FC1);
        for (int i = 0; i < number_of_images; i++)
        {
            for (int j = 0; j < n_rows * n_cols; j++)
            {
                unsigned char temp = 0;
                file.read((char *)&temp, sizeof(temp));
                float pixel_value = float((temp + 0.0) / 255.0);
                DataMat.at<float>(i, j) = pixel_value;
            }
        }
        end_time = clock();
        cost_time = (end_time - start_time) / CLOCKS_PER_SEC;
        cout << "读取Image数据完毕......" << cost_time << "s\n";
    }
    else
    {
        cout << "read mnist data fails" << endl;
    }

    file.close();
    return DataMat;
}

Mat KNN::read_mnist_label(const string fileName, const size_t num)
{
    int magic_number;
    int number_of_items;

    Mat LabelMat;

    ifstream file(fileName, ios::binary);
    if (file.is_open())
    {
        cout << "成功打开Label集 ... \n";

        file.read((char *)&magic_number, sizeof(magic_number));
        file.read((char *)&number_of_items, sizeof(number_of_items));
        magic_number = reverseInt(magic_number);
        number_of_items = num;

        start_time = clock();
        LabelMat = Mat::zeros(num, 1, CV_32SC1);
        for (int i = 0; i < num; i++)
        {
            unsigned char temp = 0;
            file.read((char *)&temp, sizeof(temp));
            LabelMat.at<unsigned int>(i, 0) = (unsigned int)temp;
        }
        end_time = clock();
        cost_time = (end_time - start_time) / CLOCKS_PER_SEC;
        cout << "读取Label数据完毕......" << cost_time << "s\n";
    }
    else
    {
        cout << "read mnist label fails" << endl;
    }
    file.close();
    return LabelMat;
}

std::vector<std::pair<float, unsigned int>>
KNN::core(const Mat &train_labels, const Mat &train_images, const Mat &test_image, const size_t num)
{
    std::vector<std::pair<float, unsigned int>> scores;
    for (int i = 0; i < num; i++)
    {
        float distance = 0.0;
        for (int j = 0; j < train_images.cols; j++)
        {
            distance += abs(test_image.at<float>(0, j) - train_images.at<float>(i, j));
        }
        scores.emplace_back(distance, train_labels.at<unsigned int>(i, 0));
    }
    sort(scores.begin(), scores.end(), std::less<std::pair<float, unsigned int>>());
    return (scores);
}

void KNN::test(const std::string &data_path, const size_t num)
{
    // load MNIST dataset
    Mat train_labels = read_mnist_label(data_path + "/train-labels.idx1-ubyte", num);
    Mat train_images = read_mnist_image(data_path + "/train-images.idx3-ubyte", num);
    Mat test_labels = read_mnist_label(data_path + "/t10k-labels.idx1-ubyte", num);
    Mat test_images = read_mnist_image(data_path + "/t10k-images.idx3-ubyte", num);

    std::vector<std::pair<float, unsigned int>> scores;
    std::cout << "print the 5 labels with highest score" << std::endl;
    int count = 0;
    for (int k = 0; k < num; k++)
    {
        scores = core(train_labels, train_images, test_images.row(k), num);
        if (k < 5)
        {
            std::cout << scores[0].second << ",  " << scores[0].first << ",  "
                      << "expected: " << test_labels.at<unsigned int>(k, 0) << std::endl;
        }
        if (scores[0].second == test_labels.at<unsigned int>(k, 0))
        {
            count++;
        }
        scores.clear();
    }
    std::cout << "!!! The success rate is " << (float)count / num << std::endl;
}

int KNN::recognize(const std::string &data_path, const std::string &filename, const size_t num)
{
    // load MNIST dataset
    Mat train_labels = read_mnist_label(data_path + "/train-labels.idx1-ubyte", num);
    Mat train_images = read_mnist_image(data_path + "/train-images.idx3-ubyte", num);

    // load picture
    Mat srcimg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if (srcimg.data == nullptr)
    {
        cout << "image read error" << endl;
        return 1;
    }
    Mat img;
    cv::resize(srcimg, img, cv::Size(28, 28));
    Mat dstimg = Mat::zeros(img.rows, img.cols, CV_32FC1);
    cv::threshold(img, dstimg, 127, 255, cv::THRESH_BINARY);
    Mat DataMat = Mat::zeros(1, img.rows * img.cols, CV_32FC1);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            dstimg.at<uchar>(i, j) = 255 - dstimg.at<uchar>(i, j);
            int k = i * img.cols + j;
            float pixel_value = float((dstimg.at<uchar>(i, j) + 0.0) / 255.0);
            DataMat.at<float>(0, k) = pixel_value;
        }
    }

    // recognize
    std::cout << "start recognizing......" << std::endl;
    int rows = num;
    std::vector<std::pair<float, unsigned int>> res;
    float distance;
    for (int i = 0; i < rows; i++)
    {
        distance = 0.0;
        for (int j = 1; j < train_images.cols; j++)
        {
            distance += (DataMat.at<float>(0, j) - train_images.at<float>(i, j)) * (DataMat.at<float>(0, j) - train_images.at<float>(i, j));
        }
        res.emplace_back(distance, train_labels.at<unsigned int>(i, 0));
    }
    sort(res.begin(), res.end(), std::less<std::pair<float, unsigned int>>());

    for (int i = 0; i < 10; i++)
    {
        std::cout << res[i].second << ",  " << res[i].first << std::endl;
    }
    return 0;
}

int KNN::ciphertext_recognize(const std::string &data_path, const std::string &filename, const size_t num)
{
    //   We start by setting up the CKKS scheme.
    EncryptionParameters parms(scheme_type::CKKS);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 60}));
    double scale = pow(2.0, 40);
    auto context = SEALContext::Create(parms);
    // keys for images
    KeyGenerator keygen(context);
    auto public_key = keygen.public_key();
    auto secret_key = keygen.secret_key();
    auto relin_keys = keygen.relin_keys();
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);
    // keys for labels
    KeyGenerator keygen2(context);
    auto public_key2 = keygen2.public_key();
    auto secret_key2 = keygen2.secret_key();
    auto relin_keys2 = keygen2.relin_keys();
    Encryptor encryptor2(context, public_key2);
    Evaluator evaluator2(context);
    Decryptor decryptor2(context, secret_key2);
    CKKSEncoder encoder2(context);

    // load MNIST dataset
    Mat train_labels = read_mnist_label(data_path + "/train-labels.idx1-ubyte", num);
    Mat train_images = read_mnist_image(data_path + "/train-images.idx3-ubyte", num);

    // load picture
    Mat srcimg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if (srcimg.data == nullptr)
    {
        cout << "image read error" << endl;
        return 1;
    }
    Mat img;
    cv::resize(srcimg, img, cv::Size(28, 28));
    Mat dstimg = Mat::zeros(img.rows, img.cols, CV_32FC1);
    cv::threshold(img, dstimg, 127, 255, cv::THRESH_BINARY);
    Mat DataMat = Mat::zeros(1, img.rows * img.cols, CV_32FC1);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            dstimg.at<uchar>(i, j) = 255 - dstimg.at<uchar>(i, j);
            int k = i * img.cols + j;
            float pixel_value = float((dstimg.at<uchar>(i, j) + 0.0) / 255.0);
            DataMat.at<float>(0, k) = pixel_value;
        }
    }
    //encrypt
    std::cout << "start encrypting......" << std::endl;
    size_t rows = num;
    std::vector<std::vector<Ciphertext>>
        encrypted_train_labels(rows, std::vector<Ciphertext>(train_labels.cols));
    std::vector<std::vector<Ciphertext>>
        encrypted_train_images(rows, std::vector<Ciphertext>(train_images.cols));
    std::vector<std::vector<Ciphertext>>
        encrypted_image(1, std::vector<Ciphertext>(DataMat.cols));
    Plaintext tmp, tmp2;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < train_images.cols; j++)
        {
            encoder.encode(train_images.at<float>(i, j), scale, tmp);
            encryptor.encrypt(tmp, encrypted_train_images[i][j]);
        }
        encoder2.encode(train_labels.at<unsigned int>(i, 0), scale, tmp2);
        encryptor2.encrypt(tmp2, encrypted_train_labels[i][0]);
    }

    for (int j = 0; j < DataMat.cols; j++)
    {
        encoder.encode(DataMat.at<float>(0, j), scale, tmp);
        encryptor.encrypt(tmp, encrypted_image[0][j]);
    }
    // recognize
    std::cout << "start recognizing......" << std::endl;
    Ciphertext ctmp;
    std::vector<std::pair<Ciphertext, Ciphertext>> encrypted_res;
    Ciphertext distance, square_dist;
    Ciphertext dsum;
    for (int i = 0; i < rows; i++)
    {
        evaluator.sub(encrypted_image[0][0], encrypted_train_images[i][0], distance);
        evaluator.square(distance, dsum);
        for (int j = 1; j < encrypted_train_images[0].size(); j++)
        {
            evaluator.sub(encrypted_image[0][j], encrypted_train_images[i][j], distance);
            evaluator.square(distance, square_dist);
            evaluator.relinearize_inplace(square_dist, relin_keys);
            evaluator.rescale_to_next_inplace(square_dist);
            parms_id_type last_parms_id = square_dist.parms_id();
            evaluator.mod_switch_to_inplace(dsum, last_parms_id);
            dsum.scale() = pow(2.0, 40);
            square_dist.scale() = pow(2.0, 40);
            evaluator.add(dsum, square_dist, dsum);
            //distance += (test_image[0][j] - train_images[i][j])^2;
        }
        encrypted_res.emplace_back(dsum, encrypted_train_labels[i][0]);
    }

    //decrypt
    std::cout << "start decrypting......" << std::endl;
    std::vector<std::pair<float, double>> res(encrypted_train_images.size());
    vector<double> dtmp;
    vector<double> dtmp2;
    for (int i = 0; i < rows; i++)
    {
        decryptor.decrypt(encrypted_res[i].first, tmp);
        encoder.decode(tmp, dtmp);
        res[i].first = dtmp[0];

        decryptor2.decrypt(encrypted_res[i].second, tmp2);
        encoder2.decode(tmp2, dtmp2);
        res[i].second = dtmp2[0] > 0 ? dtmp2[0] : 0;
    }
    sort(res.begin(), res.end(), std::less<std::pair<float, double>>());

    for (int i = 0; i < 10; i++)
    {
        std::cout << res[i].second << ",  " << res[i].first << std::endl;
    }
    return 0;
}

int KNN::ciphertext_recognize_compressed(const std::string &data_path, const std::string &filename)
{
    //   We start by setting up the CKKS scheme.
    EncryptionParameters parms(scheme_type::CKKS);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 60}));
    double scale = pow(2.0, 40);
    auto context = SEALContext::Create(parms);
    // keys for images
    KeyGenerator keygen(context);
    auto public_key = keygen.public_key();
    auto secret_key = keygen.secret_key();
    auto relin_keys = keygen.relin_keys();
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);
    // keys for labels
    KeyGenerator keygen2(context);
    auto public_key2 = keygen2.public_key();
    auto secret_key2 = keygen2.secret_key();
    auto relin_keys2 = keygen2.relin_keys();
    Encryptor encryptor2(context, public_key2);
    Evaluator evaluator2(context);
    Decryptor decryptor2(context, secret_key2);
    CKKSEncoder encoder2(context);

    // load MNIST dataset
    size_t num = encoder.slot_count();
    Mat train_labels = read_mnist_label(data_path + "/train-labels.idx1-ubyte", num);
    Mat train_images = read_mnist_image(data_path + "/train-images.idx3-ubyte", num);

    // load picture
    Mat srcimg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if (srcimg.data == nullptr)
    {
        cout << "image read error" << endl;
        return 1;
    }
    Mat img;
    cv::resize(srcimg, img, cv::Size(28, 28));
    Mat dstimg = Mat::zeros(img.rows, img.cols, CV_32FC1);
    cv::threshold(img, dstimg, 127, 255, cv::THRESH_BINARY);
    Mat DataMat = Mat::zeros(1, img.rows * img.cols, CV_32FC1);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            dstimg.at<uchar>(i, j) = 255 - dstimg.at<uchar>(i, j);
            int k = i * img.cols + j;
            float pixel_value = float((dstimg.at<uchar>(i, j) + 0.0) / 255.0);
            DataMat.at<float>(0, k) = pixel_value;
        }
    }
    //encrypt
    std::cout << "start encrypting......" << std::endl;
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;
    // input_train_images contains slot_count images. Every image is store in one column.
    std::vector<std::vector<double>> input_train_images(train_images.cols, std::vector<double>(slot_count));
    std::vector<double> input_train_labels(slot_count);

    for (size_t i = 0; i < train_images.cols; i++)
    {
        for (size_t j = 0; j < slot_count; j++)
        {
            input_train_images[i][j] = train_images.at<float>(j, i);
            input_train_labels[j] = (double)1.0 * train_labels.at<unsigned int>(j, 0);
        }
    }
    size_t rows = train_images.cols;
    Ciphertext encrypted_train_labels;
    std::vector<Ciphertext> encrypted_train_images(rows);
    vector<Ciphertext> encrypted_image(rows);
    Plaintext tmp, tmp2;
    for (int i = 0; i < rows; i++)
    {
        encoder.encode(input_train_images[i], scale, tmp);
        encryptor.encrypt(tmp, encrypted_train_images[i]);
    }
    encoder2.encode(input_train_labels, scale, tmp2);
    encryptor2.encrypt(tmp2, encrypted_train_labels);

    std::vector<std::vector<double>> input_image(train_images.cols, std::vector<double>(slot_count));
    for (int i = 0; i < slot_count; i++)
    {
        for (int j = 0; j < DataMat.cols; j++)
        {
            input_image[j][i] = DataMat.at<float>(0, j);
        }
    }
    for (int i = 0; i < train_images.cols; i++)
    {
        encoder.encode(input_image[i], scale, tmp);
        encryptor.encrypt(tmp, encrypted_image[i]);
    }

    // recognize
    std::cout << "start recognizing......" << std::endl;
    Ciphertext ctmp;
    Ciphertext distance, square_dist;
    Ciphertext dsum;
    evaluator.sub(encrypted_image[0], encrypted_train_images[0], distance);
    evaluator.square(distance, dsum);
    for (int i = 1; i < rows; i++)
    {
        evaluator.sub(encrypted_image[i], encrypted_train_images[i], distance);
        evaluator.square(distance, square_dist);
        evaluator.relinearize_inplace(square_dist, relin_keys);
        evaluator.rescale_to_next_inplace(square_dist);
        parms_id_type last_parms_id = square_dist.parms_id();
        evaluator.mod_switch_to_inplace(dsum, last_parms_id);
        dsum.scale() = pow(2.0, 40);
        square_dist.scale() = pow(2.0, 40);
        evaluator.add(dsum, square_dist, dsum);
        //distance += (test_image[0][j] - train_images[i][j])^2;
    }

    //decrypt
    std::cout << "start decrypting......" << std::endl;
    vector<double> dtmp;
    vector<double> dtmp2;
    decryptor.decrypt(dsum, tmp);
    encoder.decode(tmp, dtmp);

    decryptor2.decrypt(encrypted_train_labels, tmp2);
    encoder2.decode(tmp2, dtmp2);

    std::vector<std::pair<float, double>> res(slot_count);

    for (int i = 0; i < slot_count; i++)
    {
        res[i].first = dtmp[i];
        res[i].second = dtmp2[i] > 0 ? dtmp2[i] : 0;
    }
    sort(res.begin(), res.end(), std::less<std::pair<float, double>>());

    for (int i = 0; i < 10; i++)
    {
        std::cout << res[i].second << ",  " << res[i].first << std::endl;
    }
    return 0;
}

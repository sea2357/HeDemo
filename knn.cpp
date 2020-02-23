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

//测试item个数
int testNum = 10000;

int KNN::reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

Mat KNN::read_mnist_image(const string fileName)
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
        //cout << magic_number << " " << number_of_images << " " << n_rows << " " << n_cols << endl;

        magic_number = reverseInt(magic_number);
        number_of_images = reverseInt(number_of_images);
        n_rows = reverseInt(n_rows);
        n_cols = reverseInt(n_cols);
        cout << "MAGIC NUMBER = " << magic_number
             << " ;NUMBER OF IMAGES = " << number_of_images
             << " ; NUMBER OF ROWS = " << n_rows
             << " ; NUMBER OF COLS = " << n_cols << endl;

        //-test-
        //number_of_images = testNum;
        //输出第一张和最后一张图，检测读取数据无误
        Mat s = Mat::zeros(n_rows, n_cols, CV_32FC1);
        Mat e = Mat::zeros(n_rows, n_cols, CV_32FC1);

        cout << "开始读取Image数据......\n";
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

                //打印第一张和最后一张图像数据
                if (i == 0)
                {
                    s.at<float>(j / n_cols, j % n_cols) = pixel_value;
                }
                else if (i == number_of_images - 1)
                {
                    e.at<float>(j / n_cols, j % n_cols) = pixel_value;
                }
            }
        }
        end_time = clock();
        cost_time = (end_time - start_time) / CLOCKS_PER_SEC;
        cout << "读取Image数据完毕......" << cost_time << "s\n";

        //        imshow("first image", s);
        //        imshow("last image", e);
        //        waitKey(0);
    }
    else
    {
        cout << "read mnist data fails" << endl;
    }

    file.close();
    return DataMat;
}

Mat KNN::read_mnist_label(const string fileName)
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
        number_of_items = reverseInt(number_of_items);

        cout << "MAGIC NUMBER = " << magic_number << "  ; NUMBER OF ITEMS = " << number_of_items << endl;

        //-test-
        //number_of_items = testNum;
        //记录第一个label和最后一个label
        unsigned int s = 0, e = 0;

        cout << "开始读取Label数据......\n";
        start_time = clock();
        LabelMat = Mat::zeros(number_of_items, 1, CV_32SC1);
        for (int i = 0; i < number_of_items; i++)
        {
            unsigned char temp = 0;
            file.read((char *)&temp, sizeof(temp));
            LabelMat.at<unsigned int>(i, 0) = (unsigned int)temp;

            //打印第一个和最后一个label
            if (i == 0)
                s = (unsigned int)temp;
            else if (i == number_of_items - 1)
                e = (unsigned int)temp;
        }
        end_time = clock();
        cost_time = (end_time - start_time) / CLOCKS_PER_SEC;
        cout << "读取Label数据完毕......" << cost_time << "s\n";

        cout << "first label = " << s << endl;
        cout << "last label = " << e << endl;
    }
    else
    {
        cout << "read mnist label fails" << endl;
    }
    file.close();
    return LabelMat;
}

std::vector<std::pair<float, unsigned int>>
KNN::core(const Mat &train_labels, const Mat &train_images, const Mat &test_image)
{
    std::vector<std::pair<float, unsigned int>> scores;
    for (int i = 0; i < train_images.rows; i++)
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

void KNN::test(const std::string &data_path)
{
    //   We start by setting up the CKKS scheme.
    EncryptionParameters parms(scheme_type::CKKS);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 60}));
    double scale = pow(2.0, 40);
    auto context = SEALContext::Create(parms);
    KeyGenerator keygen(context);
    // load MNIST dataset
    Mat train_labels = read_mnist_label(data_path + "/train-labels.idx1-ubyte");
    Mat train_images = read_mnist_image(data_path + "/train-images.idx3-ubyte");
    Mat test_labels = read_mnist_label(data_path + "/t10k-labels.idx1-ubyte");
    Mat test_images = read_mnist_image(data_path + "/t10k-images.idx3-ubyte");

    auto public_key = keygen.public_key();
    auto secret_key = keygen.secret_key();
    auto relin_keys = keygen.relin_keys();
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    CKKSEncoder encoder(context);
#if 0
    std::vector<std::vector<Ciphertext>> 
        encrypted_train_labels(train_labels.rows, std::vector<Ciphertext>(train_labels.cols));
    std::vector<std::vector<Ciphertext>> 
        encrypted_train_images(train_images.rows, std::vector<Ciphertext>(train_images.cols));
    std::vector<std::vector<Ciphertext>> 
        encrypted_test_images(test_images.rows, std::vector<Ciphertext>(test_images.cols));
    Plaintext tmp;
    
    for (int i = 0; i < train_images.rows; i++)
    {
        for (int j = 0; j < train_images.cols; j++)
        {
            encoder.encode(train_images.at<float>(i, j), scale, tmp);
            encryptor.encrypt(tmp, encrypted_train_images[i][j]);
        }
        encoder.encode(train_labels.at<unsigend int>(i,0), scale, tmp);
        encryptor.encrypt(tmp, encrypted_train_labels[i][0]);
    }
#endif
    std::vector<std::pair<float, unsigned int>> scores;
    std::cout << "print the 5 labels with highest score" << std::endl;
    int count = 0;
    for (int k = 0; k < test_images.rows; k++)
    {
        scores = core(train_labels, train_images, test_images.row(k));
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
    std::cout << "!!! The success rate is " << (float)count / test_images.rows << std::endl;
}

int KNN::recognize(const std::string &data_path, const std::string &filename)
{
    //   We start by setting up the CKKS scheme.
    EncryptionParameters parms(scheme_type::CKKS);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 60}));
    double scale = pow(2.0, 40);
    auto context = SEALContext::Create(parms);
    KeyGenerator keygen(context);
    auto public_key = keygen.public_key();
    auto secret_key = keygen.secret_key();
    auto relin_keys = keygen.relin_keys();
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    CKKSEncoder encoder(context);
    // load MNIST dataset
    Mat train_labels = read_mnist_label(data_path + "/train-labels.idx1-ubyte");
    Mat train_images = read_mnist_image(data_path + "/train-images.idx3-ubyte");

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
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;
    size_t rows = 30;
    std::vector<std::vector<Ciphertext>>
        encrypted_train_labels(rows, std::vector<Ciphertext>(train_labels.cols));
    std::vector<std::vector<Ciphertext>>
        encrypted_train_images(rows, std::vector<Ciphertext>(train_images.cols));
    std::vector<std::vector<Ciphertext>>
        encrypted_image(1, std::vector<Ciphertext>(DataMat.cols));
    Plaintext tmp;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < train_images.cols; j++)
        {
            encoder.encode(train_images.at<float>(i, j), scale, tmp);
            encryptor.encrypt(tmp, encrypted_train_images[i][j]);
        }
        encoder.encode(train_labels.at<unsigned int>(i, 0), scale, tmp);
        encryptor.encrypt(tmp, encrypted_train_labels[i][0]);
    }

    for (int j = 0; j < DataMat.cols; j++)
    {
        encoder.encode(DataMat.at<float>(0, j), scale, tmp);
        encryptor.encrypt(tmp, encrypted_image[0][j]);
    }
    // recognize
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
            //distance += abs(test_image[0][j] - train_images[i][j]);
        }
        encrypted_res.emplace_back(dsum, encrypted_train_labels[i][0]);
    }

    //decrypt
    std::vector<std::pair<float, unsigned int>> res(encrypted_train_images.size());
    vector<double> dtmp;
    for (int i = 0; i < encrypted_train_images.size(); i++)
    {
        decryptor.decrypt(encrypted_res[i].first, tmp);
        encoder.decode(tmp, dtmp);
        res[i].first = dtmp[0];

        decryptor.decrypt(encrypted_res[i].second, tmp);
        encoder.decode(tmp, dtmp);
        res[i].second = dtmp[0];
    }
    sort(res.begin(), res.end(), std::less<std::pair<float, unsigned int>>());

    for (int i = 0; i < 10; i++)
    {
        std::cout << res[i].second << ",  " << res[i].first << std::endl;
    }
    return 0;
}

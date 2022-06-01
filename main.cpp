//<Junwei Zhou>
#include <Eigen/Dense>
#include <eigen3/Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <pthread.h>

using namespace std;
using namespace cv;
using namespace Eigen;

#define MAX_THREAD 4
#define MAX_ID 400

Mat Analy, Left, Right;
double M_l[3][3] = {{0.299, 0.587, 0.114}, {0, 0, 0}, {0, 0, 0}};
double M_r[3][3] = {{0, 0, 0}, {0, 0, 0}, {0.299, 0.587, 0.114}};
//Mat A = (Mat_<float>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
typedef Point3_<uint8_t> Pixel;
int Col_i;

//Matrix3f M_l << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
//int Col_i;

//Mat M_l(3,3,double);
//Mat M_r = (Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);


//double M_l[3][3] = {{0,0,0}};
//double M_r[3][3];
/*
for multithreading
void *multiply(void *arg)
{

}
*/
/*
Mat analy_gen(const Mat &left_graph, const Mat &right_graph, Matrix3f M_l, Matrix3f M_r)
{
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatrixXf;
    MatrixXf left_RGB, right_RGB, analy_RGB, A, B;

    Mat left, right, analy_graph;

    // crop image to the same size
    int Row = min(right_graph.rows, left_graph.rows);
    int Col = min(right_graph.cols, left_graph.cols);
    left = left_graph(Range(0, Row), Range(0, Col));
    right = right_graph(Range(0, Row), Range(0, Col));

    int n_channels = min(left.channels(), right.channels()); // assume two pictures have same channels

    // get eigen matrix
    left = left.reshape(1, 0);
    cv2eigen(left, left_RGB);
    cout << "left size: " << left_RGB.size() << " " << left_RGB.rows() << " " << left_RGB.cols() << endl;
    cout << "left: " << left_RGB(0, seqN(0, 10)) << endl;
    left_RGB = left_RGB.reshaped(3, left_RGB.cols() * left_RGB.rows() / 3).eval(); // assume 3 channels
    cout << "left: " << left_RGB(0, seqN(0, 10)) << endl;

    right = right.reshape(1, 0);
    cv2eigen(right, right_RGB);
    right_RGB = right_RGB.reshaped(3, right_RGB.cols() * right_RGB.rows() / 3).eval(); // assume 3 channels

    // generate analygraph in matrix
    // future multithreading update
    A = M_l * left_RGB;
    cout << "A: " << A(0, seqN(0, 10)) << endl;
    B = M_r * right_RGB;
    analy_RGB = A + B;
    analy_RGB.resize(Row, analy_RGB.size()/Row);

    //cout << analy_RGB(0, seqN(0, 20)) << endl;

    // convert matrix to mat
    eigen2cv(analy_RGB, analy_graph);
    analy_graph = analy_graph.reshape(n_channels, 0);
    return analy_graph;
}
*/
void *multiply(void* arg)
{
    int i = Col_i +1;
    Pixel pixel;
    for (int j = 0; j < 3; j++)
    {
        for (int k = 0; k<3; k++)
        {
            pixel = M_l[j][k] * Left.at<Pixel>(k, j) + M_r[j][k] * Right.at<Pixel>(k, j);
        }
        // Pixel pixel = img1.at<Pixel>(0, j) * num.row(i).col(0) + img1.at<Pixel>(1, j) * num.row(i).col(1) + img1.at<Pixel>(2, j) * num.row(i).col(2);
        Analy.at<Pixel>(j, i) = pixel;
    }
}
int main(int argc, char **argv)
{
    int opt;
    int half_width = 0;
    char *image_path = NULL;
    int row, col;

    //double number[3] = {0.5, 0.5, 0.5};

    Mat left, right;
    Mat img, img1, img3;
    Mat channels[3];
    Mat r,g,b;


    left = imread("images/left.jpg", IMREAD_COLOR);
    cout << left.size() << endl;
    cvtColor(left, left, COLOR_BGR2RGB);
    right = imread("images/right.jpg", IMREAD_COLOR);
    cvtColor(right, right, COLOR_BGR2RGB);

    row = min(right.rows, left.rows);
    col = min(right.cols, left.cols);

    left = left(Range(0, row), Range(0, col));
    right = right(Range(0, row), Range(0, col));

    //Left = Mat::zeros(3, row*col, CV_8UC3);
    //Right = Mat::zeros(3, row*col, CV_8UC3);
    Analy = Mat::zeros(left.size(), CV_8UC3);

    /*cout << "left: " << left.size() << " " << left.channels() << endl;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j<10; j++)
            cout <<  left.row(i).col(j) << " ";
        cout << endl;
    }*/
    //Mat img2[3];

    //left = left.reshape(1,0);
    split(left, channels);
    r = channels[0].reshape(1, left.rows * left.cols);
    g = channels[1].reshape(1, left.rows * left.cols);
    b = channels[2].reshape(1, left.rows * left.cols);
    hconcat(r, g, Left);
    hconcat(Left, b, Left);

    split(right, channels);
    r = channels[0].reshape(1, right.rows * right.cols);
    g = channels[1].reshape(1, right.rows * right.cols);
    b = channels[2].reshape(1, right.rows * right.cols);
    hconcat(r, g, Right);
    hconcat(Right, b, Right);

    Mat RGB = Mat::zeros(Left.size(), CV_8UC1);
    cout << RGB.size() << " " << RGB.channels() << endl;
    cout << RGB.rows << " " << RGB.cols << endl;

    Pixel pixel;

    for (int k = 0; k < RGB.rows; k++)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
            pixel = M_l[i][j] * Left.at<Pixel>(k, j) + M_r[i][j] * Right.at<Pixel>(k, j);
            }
        // Pixel pixel = img1.at<Pixel>(0, j) * num.row(i).col(0) + img1.at<Pixel>(1, j) * num.row(i).col(1) + img1.at<Pixel>(2, j) * num.row(i).col(2);
            RGB.at<Pixel>(k, i) = pixel;
        }
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 10; j++) {
            cout << RGB.col(i).row(j) << " ";
        }
        cout << endl;
    }
    
    Mat R(Analy.size(), CV_8UC1);
    Mat G(Analy.size(), CV_8UC1);
    Mat B(Analy.size(), CV_8UC1);

    channels[0] = r.reshape(1, row);
    channels[1] = g.reshape(1, row);
    channels[2] = b.reshape(1, row);

    RGB.col(0).copyTo(r);
    R= r.reshape(1, row);
    RGB.col(1).copyTo(g);
    G = g.reshape(1, row);
    RGB.col(2).copyTo(b);
    B = b.reshape(1, row);
    //merge(channels, Analy);
    //merge(R, G, B, Analy);
    //Mat G(Left.size(), CV_8UC1);
    //Mat B(Left.size(), CV_8UC1);
    //R = RGB.col(0);
    //cout << RGB.rows<< " " << RGB.cols << endl;
    //RGB.col(0).copyTo(r);
    //RGB.col(0).copyTo(r);
    //cout << "r: " << r.rows << " " << r.cols << endl;
    //cout << "Left: " << Left.size() << " " << Left.rows << " " << Left.cols << " " << Left.channels() << endl;

    //R = r.reshape(1, row);
    //cout << r.isContinuous() << " " << R.isContinuous() << endl;
    //cout << "r: " << r.size() << " " << r.channels() << endl;
    //Mat R = r.reshape(Left.rows, Left.cols);
    //cout << "r: " << r.size() << " " << r.channels() << endl;
    merge(channels, 3, Analy);
    cout << Analy.size() << endl;


        for (int j = 0; j < 10; j++)
        {
            cout << Analy.col(0).row(j) << " ";
        }
        cout << endl;


    //r = R.reshape(Left.rows, Left.cols);
    //cout << "r: " << r.size() << " " << r.channels() << endl;
    // Mat R = r.reshape(Left.size());
    // g = RGB.col(1);
    // Mat G = g.reshape(Left.size());
    // b = RGB.col(2);
    // Mat B = b.reshape(Left.size());
    // cout << "R: " << R.size() << " " << R.channels() << endl;
    
    /*
    pthread_t threads[MAX_THREAD];

    // Creating four threads, each evaluating its own part
    for (i = 0; i < MAX_THREAD; i++)
    {
        int *p;
        pthread_create(&threads[i], NULL, multiply, (void *)(p));
    }

    // joining and waiting for all threads to complete
    for (i = 0; i < MAX_THREAD; i++)
        pthread_join(threads[i], NULL);
    */

    //imshow("R", R);
    //imshow("G", G);
    //imshow("B", B);
    cvtColor(Analy, Analy, COLOR_RGB2BGRA);
    imshow("Analygraph", Analy / 255.0);
    imwrite("Analygraph.jpg", Analy);
    waitKey(0);
    destroyAllWindows();
    
    return 0;

    //hconcat(left_ch[0], left_ch[1], Left);
    //hconcat(Left, left_ch[2], Left);
    
    /*cout << "Left: " << Left.size() << " " << Left.channels() << endl;
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 10; j++)
            cout << Left.col(i).row(j) << " ";
        cout << endl;
    }*/
    //split(right, right_ch);
    //hconcat(right_ch[0], right_ch[1], Right);
   // hconcat(Right, right_ch[2], Right);
    
    
    // cout << "img2: " << img2.size() << " " << img2.channels() << endl;
    //Mat A = img2[0];
    //Mat B = img2[1];
    //Mat C = img2[2];
    //hconcat(A, B, img3);
    //hconcat(img3, C, img3);
    /*&
    img = imread("images/pic.jpg", IMREAD_COLOR);
    cout << "img: " << img.size() << " " << img.channels() << endl;
    for (i = 0; i < 10; i++)
    {
        cout << img.row(0).col(i) << " ";
    }
    cout << endl;
    Mat img2[3];
    split(img, img2);
    // cout << "img2: " << img2.size() << " " << img2.channels() << endl;
    Mat A = img2[0];
    Mat B = img2[1];
    Mat C = img2[2];
    hconcat(A, B, img3);
    hconcat(img3, C, img3);
    cout << "img3: " << img3.size() << " " << img3.channels() << endl;
    for (i = 0; i < 10; i++)
    {
        cout << img3.row(0).col(i) << " ";
    }
    cout << endl;
    */
    /*
    cout << "img: " << img.size() << " " << img.channels() << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << img.row(0).col(i) << " ";
    }
    cout << endl;

    img1 = img.reshape(1,0);
    cout << "img: " << img1.size() << " " << img1.channels() << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << img1.row(0).col(i) << " ";
    }
    cout << endl;

    Mat img2[3];
    split(img, img2);
    //cout << "img2: " << img2.size() << " " << img2.channels() << endl;
    Mat A = img2[0];
    Mat B = img2[1];
    Mat C = img2[2];
    hconcat(A, B, img3);
    hconcat(img3, C, img3);

    for (int i = 0; i < 10; i++)
    {
        cout << A.row(0).col(i) << " ";
    }
    cout << endl;

    for (int i = 0; i < 10; i++)
    {
        cout << img3.row(0).col(i) << " ";
    }
    cout << endl;

    img1 = img.reshape(3, img.rows*img.cols*img.channels()/3);

    Mat num = (Mat_<float>(3, 3) << 0.3185, 0.0769, 0.0109, 0.1501, 0.0767, 0.0056, 0.0007, 0.0020, 0.0156);
    Mat num2 = Mat::zeros(img.size(), CV_8UC3);

    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < img1.rows; j++)
        {
            Pixel pixel = img1.at<Pixel>(0, j) * number[0] + img1.at<Pixel>(1, j) * number[1] + img1.at<Pixel>(2, j) * number[2];
            //Pixel pixel = img1.at<Pixel>(0, j) * num.row(i).col(0) + img1.at<Pixel>(1, j) * num.row(i).col(1) + img1.at<Pixel>(2, j) * num.row(i).col(2);
            num2.at<Pixel>(i,j) = pixel;
        }
    }
    */
    //imshow("Analygraph", Analy);
    //waitKey(0);
    //destroyAllWindows();
    //Mat num1 = num * img1;

    //Mat num2;
    //num2 = img.reshape(3, 0);

    //Mat out = 

    
    /*for (i=0; i < A.rows; i++)
    {
        for (j = 0; j < A.cols; j++)
        {
            Pixel px = A.at<Pixel>(i, j);
            cout << px << " ";
            // cout << A.row(i).col(j);
        }
        cout << endl;
    }*/

    //M_l[0] = {0, 0, 0};

    //int a[4] = {0, 0, 0, 0};
    //int x[3][2] = {1,2,3,4,5,6};
    //x[3][2] = {0,0,0,0,0,0};

    //Mat A = (Mat_<float>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
    //Mat C[3];
    //split(A, C);
    //cout << C << endl;
    //Mat A[3];
    //A = (Mat_<float>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
    //<< 1, 1, 1, 1, 1, 1, 1, 1, 1;

    //A[0] = {1,2,3};
    //M_l = Matrix3f::Zero();
    //M_r = Matrix3f::Zero();
    //Matrix3f M_l = Matrix3f::Zero();
    //Matrix3f M_r = Matrix3f::Zero();

    //M_l = {}
    /*
    while ((opt = getopt(argc, argv, "l:r:i:TGCHODR")) != -1)
    {
        switch (opt)
        {
            case 'l':
                image_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(image_path, optarg, strlen(optarg) + 1);

                left = imread(image_path, IMREAD_COLOR);
                if (!left.data)
                {
                    cout << "Error loading image." << endl;
                    return -1;
                }

                Mat B[3];
                split(left, B);
                Mat D = B.mul(C);
                //float C = A.col(0)*B[0];

                cvtColor(left, left, COLOR_BGR2RGB);

                break;
            case 'r' :
                image_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(image_path, optarg, strlen(optarg) + 1);

                right = imread(image_path, IMREAD_COLOR);
                if (!right.data)
                {
                    cout << "Error loading image." << endl;
                    return -1;
                }

                cvtColor(right, right, COLOR_BGR2RGB);
                break;
            case 'i':
                image_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(image_path, optarg, strlen(optarg) + 1);

                img = imread(image_path, IMREAD_COLOR);
                if (!img.data)
                {
                    cout << "Error loading image." << endl;
                    return -1;
                }

                cvtColor(img, img, COLOR_BGR2RGB);
                half_width = (int)img.size().width / 2;
                left = img(Range(0, img.size().height), Range(0, half_width));
                right = img(Range(0, img.size().height), Range(half_width, img.size().width));

                break;
            
            case 'T': // True analygraph
                //(Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
                //M_l.row(0) << 0.299, 0.587, 0.114;
                //M_l.row(0) = {0.299, 0.587, 0.114};
                //M_r.row(2) = (Mat_<double>(1, 3) << 0.299, 0.587, 0.114);

                cout << M_l << endl;
                cout << M_r << endl;

                break;
            
            case 'G': // Gray analygraph
                M_l.row(0) << 0.299, 0.587, 0.114;
                M_r.row(1) << 0.299, 0.587, 0.114;
                M_r.row(2) << 0.299, 0.587, 0.114;
                break;
            
            case 'C': // Color analygraph
                M_l(0, 0) = 1;
                M_r(1, 1) = 1;
                M_r(2, 2) = 1;
                break;
            case 'H': // Half-color analygraph
                M_l.row(0) << 0.299, 0.587, 0.114;
                M_r(1, 1) = 1;
                M_r(2, 2) = 1;
                break;
            case 'O': // 3DTV-optimized analygraph
                M_l(0, 1) = 0.7;
                M_l(0, 2) = 0.3;
                M_r(1, 1) = 1;
                M_r(2, 2) = 1;
                break;
            case 'D': // DuBois analygraph
                M_l << 0.437, 0.449, 0.164, -0.062, -0.062, -0.024, -0.048, -0.050, -0.017;
                M_r << -0.011, -0.032, -0.007, 0.377, 0.761, 0.009, -0.026, -0.093, 1.234;
                break;
            case 'R': // Roscolux analygraph
                M_l << 0.3185, 0.0769, 0.0109, 0.1501, 0.0767, 0.0056, 0.0007, 0.0020, 0.0156;
                M_r << 0.0174, 0.0484, 0.1402, 0.0184, 0.1807, 0.0458, 0.0286, 0.0991, 0.7662;
                break;
            
            default:
                cout << "Unknown option character " << endl;
                return -1;
        }
    }
    */
    //out = analy_gen(left, right, M_l, M_r);
    /*
    imshow("RGB", out / 255.0);
    cvtColor(out, out, COLOR_RGB2BGRA);

    imwrite("analygraph.jpg", out);
    imshow("BGR", out / 255.0);
    waitKey(0);
    destroyAllWindows();

    */
    //return 0;
}
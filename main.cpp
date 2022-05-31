#include <Eigen/Dense>
#include <eigen3/Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "analygen.h"

#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>


using namespace std;
using namespace cv;
using namespace Eigen;

#define DIM 3
#define MAX_THREAD 4
double Ml[DIM][DIM];
double Mr[DIM][DIM];


Mat analy_gen(const Mat &left_graph, const Mat &right_graph, Matrix3f M_l, Matrix3f M_r)
{
    //Mat analygraph = 0.5*left + 0.5*right;
    //typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
    MatrixXd left_RGB, right_RGB, analy_RGB, A, B;
    //typedef Matrix<double, 3, Dynamic> Eigen::Matrix3Xd
    //Matrix3Xd left_RGB, right_RGB, analy_RGB, A, B;

    Mat left, right, analy_graph;
    //MatrixXf mat(Dynamic, Dynamic);

    //Matrix<double, 3, Dynamic> analy_RGB;
    //Matrix<double, 3, Dynamic> left_RGB;
   // Matrix<double, 3, Dynamic> right_RGB;
    //int Row = min(right_graph.size().width, left_graph.size().width);
    //int Col = min(right_graph.size().height, left_graph.size().height);

    cout << "left: " << left_graph.size() << " " << left_graph.channels() << endl;
    cout << "right: " << right_graph.size() << " " << right_graph.channels() << endl;

    int Row = min(right_graph.rows, left_graph.rows);
    int Col = min(right_graph.cols, left_graph.cols);

    cout << "Row:" << Row << " " << "Col:" << Col << endl;

    left = left_graph(Range(0, Row), Range(0, Col));
    right = right_graph(Range(0, Row), Range(0, Col));

    int n_channels = min(left.channels(), right.channels()); // assume two pictures have same channels

    cout << "left: " << left.size() << " " << left.channels() << endl;
    Mat img = left.reshape(1,0);
    cout << "img: " << img.size() << " " << img.channels() << endl;

    cv2eigen(img, left_RGB);
    //left_RGB.resize(3, 0); // assume 3 channels
    left_RGB.resize(3, left_RGB.cols() * left_RGB.rows() / 3); // assume 3 channels

    cout << "left_RGB: " << left_RGB.rows() << " " << left_RGB.cols() << endl;

    cout << "right: " << right.size() << " " << right.channels() << endl;
    img = right.reshape(1,0);
    cout << "img: " << img.size() << " " << img.channels() << endl;
    cv2eigen(img, right_RGB);
    //right_RGB.resize(3, 0); // assume 3 channels
    right_RGB.resize(3, right_RGB.cols() * right_RGB.rows() / 3); // assume 3 channels

    A = M_l * left_RGB;
    B = M_r * right_RGB;
    cout << "A: " << A.rows() << " " << A.cols() << endl;
    cout << "B: " << B.rows() << " " << B.cols() << endl;

    //imshow("A", eigen2cv(A));
    // imshow("left", left_channels[0]);
    // imshow("right", right);

    // waitKey(0);
    // destroyAllWindows();
    // cv2eigen(left_channels, left_RGB);

    analy_RGB = A;
    //analy_RGB = A + B;
    cout << "analy_RGB: " << analy_RGB.rows() << " " << analy_RGB.cols() << endl;
    //analy_RGB.resize(3, analy_RGB.cols() * analy_RGB.rows() / 3);
    eigen2cv(analy_RGB, analy_graph);
    cout << "analy_graph: " << analy_graph.size() << " " << analy_graph.channels() << endl;
    //analy_graph = analy_graph.reshape(Row, Col);
    analy_graph = analy_graph.reshape(n_channels, 0);
    cout << "analy_graph: " << analy_graph.size() << " " << analy_graph.channels() << endl;

    //analy_graph = analy_graph.reshape(n_channels, 0);
    //cout << left.size().width << " " << left.size().height << endl;
    //Mat left_channels[3];
    // Mat right_channels[3];
    //split(left, left_channels);

    //cv2eigen(left, left_RGB);
    //cout << left_channels[0].size().width << " " << left_channels[0].size().height << endl;
    //cv2eigen(left_channels[0], left_RGB.row(0));
    //cv2eigen(left_channels, left_RGB);
    //left_RGB.row(0) = left_channels[0];
    //left_RGB.row(1) = left_channels[1];
    //left_RGB.row(2) = left_channels[2];
    // split(right, right_channels);



    //imshow("graph", analy_graph);
    //imshow("left", left_channels[0]);
    //imshow("right", right);

    //waitKey(0);
    //destroyAllWindows();
    //cv2eigen(left_channels, left_RGB);
    /*
    cv2eigen(right, right_RGB);

    cout << left_RGB.rows() << " " << left_RGB.cols() << endl;
    cout << right_RGB.rows() << " " << right_RGB.cols() << endl;

    analy_RGB = M_l * left_RGB + M_r * right_RGB;

    
    eigen2cv(analy_RGB, analygraph);

    //Mat left_channels[3];
    //Mat right_channels[3];
    //split(left, left_channels);
    //split(right, right_channels);
    //analy_RGB.row() =

    //image.at<Vec3b>(j, i)[0];
   // Mat analygraph;
    //Mat analygraph = {left_channels[0], left_channels[1], left_channels[2]};
    //merge(left_channels, 3, analygraph);

    // cout << (left)
    */
    return analy_graph;
}

//! [includes]
int main(int argc, char **argv)
{
    int opt;
    int half_width = 0;
    char *image_path = NULL;

    Mat left, right, out, img;

    Matrix3f M_l = Matrix3f::Zero();
    Matrix3f M_r = Matrix3f::Zero();

    while ((opt = getopt(argc, argv, "l:r:i:o:TGCHODR")) != -1)
    {
        switch (opt)
        {
            case 'l':
                image_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(image_path, optarg, strlen(optarg) + 1);
                //cout << image_path << endl;
                left = imread(image_path, IMREAD_COLOR);
                if (!left.data)
                {
                    cout << "Error loading image." << endl;
                    return -1;
                }
                cvtColor(left, left, COLOR_BGR2RGB);
                //left = imread(image_path, IMREAD_COLOR);
                break;
            case 'r' :
                image_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(image_path, optarg, strlen(optarg) + 1);
                //cout << image_path << endl;
                right = imread(image_path, IMREAD_COLOR);
                if (!left.data)
                {
                    cout << "Error loading image." << endl;
                    return -1;
                }
                cvtColor(left, left, COLOR_BGR2RGB);
                break;
            case 'i':
                image_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(image_path, optarg, strlen(optarg) + 1);
                //cout << image_path << endl;
                img = imread(image_path, IMREAD_COLOR);
                if (!left.data)
                {
                    cout << "Error loading image." << endl;
                    return -1;
                }

                cvtColor(img, img, COLOR_BGR2RGB);
                half_width = (int)img.size().width / 2;
                left = img(Range(0, img.size().height), Range(0, half_width));
                right = img(Range(0, img.size().height), Range(half_width, img.size().width));
                //gen.mix = 0.5 * left + 0.5 * right;
                break;
            /*
            case 'o':
                out_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(out_path, optarg, strlen(optarg) + 1);
                break;
            */
           
            case 'T':
                M_l.row(0) << 0.299, 0.587, 0.114;
                M_r.row(2) << 0.299, 0.587, 0.114;
                //cout << M_l << endl;
                //cout << M_r << endl;
                break;
            case 'G':
                M_l.row(0) << 0.299, 0.587, 0.114;
                M_r.row(1) << 0.299, 0.587, 0.114;
                M_r.row(2) << 0.299, 0.587, 0.114;
                cout << M_l << endl;
                cout << M_r << endl;
                break;
            case 'C':
                M_l(0, 0) = 1;
                M_r(1, 1) = 1;
                M_r(2, 2) = 1;
                cout << M_l << endl;
                cout << M_r << endl;
                break;
            case 'H':
                M_l.row(0) << 0.299, 0.587, 0.114;
                M_r(1, 1) = 1;
                M_r(2, 2) = 1;
                cout << M_l << endl;
                cout << M_r << endl;
                break;
            case 'O':
                M_l(0, 1) = 0.7;
                M_l(0, 2) = 0.3;
                M_r(1, 1) = 1;
                M_r(2, 2) = 1;
                cout << M_l << endl;
                cout << M_r << endl;
                break;
            case 'D':
                M_l << 0.437, 0.449, 0.164, -0.062, -0.062, -0.024, -0.048, -0.050, -0.017;
                M_r << -0.011, -0.032, -0.007, 0.377, 0.761, 0.009, -0.026, -0.093, 1.234;
                cout << M_l << endl;
                cout << M_r << endl;
                break;
            case 'R':
                M_l << 0.3185, 0.0769, 0.0109, 0.1501, 0.0767, 0.0056, 0.0007, 0.0020, 0.0156;
                M_r << 0.0174, 0.0484, 0.1402, 0.0184, 0.1807, 0.0458, 0.0286, 0.0991, 0.7662;
                cout << M_l << endl;
                cout << M_r << endl;
                break;
            
            /*
            case '?':
                fprintf(stderr, "Unknown option character `\\x%x'.\n", opt);
                
                
                if (optopt == 'c')
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint(optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                return 1;
                
               break;
               */
            default:
                cout << "Unknown option character " << endl;
                //printf("invalid\n");
                return -1;
        }
    }

    out = analy_gen(left, right, M_l, M_r);
    cout << "out: " << out.size() << " " << out.channels() << endl;

    imwrite("analygraph.jpg", out * (1.0 / 255));
    imshow("out", out);
    waitKey(0);
    destroyAllWindows();

    //waitKey(0);
    //destroyAllWindows();

    //cout << gen.left_path << endl;
    //cout << gen.right_path << endl;
    //cout << gen.in_path << endl;
    //cout << gen.out_path << endl;
    /*
    //! [imread]
    // std::string image_path = samples::findFile("lantern.jpeg");
    std::string image_path = "images/lantern.jpeg";
    std::cout << image_path;
    Mat img = imread(image_path, IMREAD_COLOR);
    // Mat img = imread("test.jpg");
    cout << "Width : " << img.size().width << endl;
    cout << "Height: " << img.size().height << endl;
    cout << "Channels: :" << img.channels() << endl;

    int half_width = (int)img.size().width / 2;
    Mat another = img(Range(0, img.size().height), Range(0, img.size().width));
    Mat left = img(Range(0, img.size().height), Range(0, half_width));
    Mat right = img(Range(0, img.size().height), Range(half_width, img.size().width));
    Mat mix = 0.5 * left + 0.5 * right;
    // Mat cropped_image = img(Range(80, 280), Range(150, 330));
    imshow(" Original Image", another);
    imshow("Left", left);
    imshow("Right", right);
    imshow("Mix", mix);
    // imwrite("left.jpg", left);
    // imwrite("right.jpg", right);

    // 0 means loop infinitely
    waitKey(0);
    destroyAllWindows();

    
    Mat img = imread(image_path, IMREAD_COLOR);
    //! [imread]

    //! [empty]
    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    //! [empty]

    //! [imshow]
    imshow("Display window", img);
    std::cout << img;
    int k = waitKey(0); // Wait for a keystroke in the window
    //! [imshow]

    //! [imsave]
    std::cout << k;
    if (k == 's')
    {
        imwrite("starry_night.png", img);
    }
    //! [imsave]
    */
    return 0;
}
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

//#define DIM 3
//#define MAX_THREAD 4
//double Ml[DIM][DIM];
//double Mr[DIM][DIM];

Mat analy_gen(const Mat &left_graph, const Mat &right_graph, Matrix3f M_l, Matrix3f M_r)
{
    //Mat analygraph = 0.5*left + 0.5*right;
    //typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatrixXf;
    MatrixXf left_RGB, right_RGB, analy_RGB, A, B;
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
    left = left.reshape(1, 0);
    //Mat img = left.reshape(1,0);
    cout << "left: " << left.size() << " " << left.channels() << endl;

    cv2eigen(left, left_RGB);
    //left_RGB.resize(3, 0); // assume 3 channels
    //left_RGB.resize(3, left_RGB.cols() * left_RGB.rows() / 3); // assume 3 channels
    cout << "left_RGB: " << left_RGB.rows() << " " << left_RGB.cols() << endl;
    cout << left_RGB(0,seqN(0,10)) << endl;
    left_RGB.resize(3, left_RGB.cols() * left_RGB.rows() / 3); // assume 3 channels

    cout << "left_RGB: " << left_RGB.rows() << " " << left_RGB.cols() << endl;

    cout << "right: " << right.size() << " " << right.channels() << endl;
    right = right.reshape(1, 0);
    cout << "right: " << right.size() << " " << right.channels() << endl;
    cv2eigen(right, right_RGB);
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

    analy_RGB = A + B;
    //analy_RGB = A + B;
    cout << "analy_RGB: " << analy_RGB.size() << endl;
    analy_RGB.resize(Row, analy_RGB.size()/Row);
    cout << analy_RGB(0, seqN(0, 20)) << endl;
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

                left = imread(image_path, IMREAD_COLOR);
                if (!left.data)
                {
                    cout << "Error loading image." << endl;
                    return -1;
                }
                cvtColor(left, left, COLOR_BGR2RGB);

                break;
            case 'r' :
                image_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(image_path, optarg, strlen(optarg) + 1);

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

                break;
            /*
            case 'o':
                out_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(out_path, optarg, strlen(optarg) + 1);
                break;
            */
           
            case 'T': // True analygraph
                M_l.row(0) << 0.299, 0.587, 0.114;
                M_r.row(2) << 0.299, 0.587, 0.114;
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

    out = analy_gen(left, right, M_l, M_r);
    cout << "out: " << out.size() << " " << out.channels() << endl;

    imwrite("analygraph.jpg", out);
    imshow("out", out / 255.0);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
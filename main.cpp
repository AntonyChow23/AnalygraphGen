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

Mat Analy;
Matrix3f M_l;
Matrix3f M_r;
int Col_i;
/*
for multithreading
void *multiply(void *arg)
{

}
*/

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
    left_RGB.resize(3, left_RGB.cols() * left_RGB.rows() / 3); // assume 3 channels

    right = right.reshape(1, 0);
    cv2eigen(right, right_RGB);
    right_RGB.resize(3, right_RGB.cols() * right_RGB.rows() / 3); // assume 3 channels

    // generate analygraph in matrix
    // future multithreading update
    A = M_l * left_RGB;
    B = M_r * right_RGB;
    analy_RGB = A + B;
    analy_RGB.resize(Row, analy_RGB.size()/Row);

    cout << analy_RGB(0, seqN(0, 20)) << endl;

    // convert matrix to mat
    eigen2cv(analy_RGB, analy_graph);
    analy_graph = analy_graph.reshape(n_channels, 0);
    return analy_graph;
}

int main(int argc, char **argv)
{
    int opt;
    int half_width = 0;
    char *image_path = NULL;

    Mat left, right, out, img;

    M_l = Matrix3f::Zero();
    M_r = Matrix3f::Zero();
    //Matrix3f M_l = Matrix3f::Zero();
    //Matrix3f M_r = Matrix3f::Zero();

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

    imwrite("analygraph.jpg", out);
    imshow("out", out / 255.0);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
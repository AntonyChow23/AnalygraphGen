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
#include <mutex>

#include <chrono>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace chrono;

#define MAX_THREAD 1

Mat Left, Right, RGB;
double M_l[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
double M_r[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
typedef Point3_<uint8_t> Pixel;
mutex mu;

//argument to thread
typedef struct thread_args
{
    int start;
    int end;
} thargs_t;

//Provide start and end column index, compute multiplication on these columns
void *multiply(void* arg)
{
    thargs_t *thargs = (thargs_t*)arg;
    if (thargs == NULL)
        pthread_exit(NULL);
    for (int k = thargs->start; k < thargs->end; k++)
    {
        for (int i = 0; i < 3; i++)
        {
            RGB.row(k).col(i) = M_l[i][0] * Left.row(k).col(0) + M_l[i][1] * Left.row(k).col(1) + M_l[i][2] * Left.row(k).col(2);
            RGB.row(k).col(i) += M_r[i][0] * Right.row(k).col(0) + M_r[i][1] * Right.row(k).col(1) + M_r[i][2] * Right.row(k).col(2);
        }
    }
    pthread_exit(NULL);
    return NULL;
}
int main(int argc, char **argv)
{
    auto start_time = high_resolution_clock::now();

    int opt, row, col;
    int half_width = 0;
    char *image_path = NULL;

    Mat left, right, Analy, img, r, g, b;
    Mat channels[3];


    while ((opt = getopt(argc, argv, "l:r:i:TGCHODR")) != -1)
    {
        switch (opt)
        {
        case 'l':// left image
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
        case 'r'://right image
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
        case 'i':// single image
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
            M_l[0][0] = 0.299; M_l[0][1] = 0.587; M_l[0][2] = 0.114;
            M_r[2][0] = 0.299; M_r[2][1] = 0.587; M_r[2][2] = 0.114;
            break;

        case 'G': // Gray analygraph
            M_l[0][0] = 0.299; M_l[0][1] = 0.587; M_l[0][2] = 0.114;
            M_r[1][0] = 0.299; M_r[1][1] = 0.587; M_r[1][2] = 0.114;
            M_r[2][0] = 0.299; M_r[2][1] = 0.587; M_r[2][2] = 0.114;
            break;

        case 'C': // Color analygraph
            M_l[0][0] = 1;
            M_r[1][1] = 1; M_r[2][2] = 1;
            break;
        case 'H': // Half-color analygraph
            M_l[0][0] = 0.299; M_l[0][1] = 0.587; M_l[0][2] = 0.114;
            M_r[1][1] = 1; M_r[2][2] = 1;
            break;
        case 'O': // 3DTV-optimized analygraph
            M_l[0][1] = 0.7; M_l[0][2] = 0.3;
            M_r[1][1] = 1; M_r[2][2] = 1;
            break;
        case 'D': // DuBois analygraph
            M_l[0][0] = 0.437; M_l[0][1] = 0.449; M_l[0][2] = 0.164;
            M_l[1][0] = -0.062; M_l[1][1] = -0.062; M_l[1][2] = -0.024;
            M_l[2][0] = -0.048; M_l[2][1] = -0.050; M_l[2][2] = -0.017;

            M_r[0][0] = -0.011; M_r[0][1] = -0.032; M_r[0][2] = -0.007;
            M_r[1][0] = 0.377; M_r[1][1] = 0.761; M_r[1][2] = 0.009;
            M_r[2][0] = -0.026; M_r[2][1] = -0.093; M_r[2][2] = 1.234;
            break;
        case 'R': // Roscolux analygraph
            M_l[0][0] = 0.3185; M_l[0][1] = 0.0769; M_l[0][2] = 0.0109;
            M_l[1][0] = 0.1501; M_l[1][1] = 0.0767; M_l[1][2] = 0.0056;
            M_l[2][0] = 0.007; M_l[2][1] = 0.0020; M_l[2][2] = 0.0156;

            M_r[0][0] = 0.0174; M_r[0][1] = 0.0484; M_r[0][2] = 0.1402;
            M_r[1][0] = 0.0184; M_r[1][1] = 0.1807; M_r[1][2] = 0.0458;
            M_r[2][0] = 0.0286; M_r[2][1] = 0.0991; M_r[2][2] = 0.7662;
            break;

        default:
            cout << "Unknown option character " << endl;
            return -1;
        }
    }

    // make left and right have same size
    row = min(right.rows, left.rows);
    col = min(right.cols, left.cols);
    left = left(Range(0, row), Range(0, col));
    right = right(Range(0, row), Range(0, col));
    Analy = Mat::zeros(left.size(), CV_8UC3);

    //split RGB channels and rearrange them
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
    RGB = Mat::zeros(Left.size(), CV_8UC1);

    // can use pointer instead of fixsized array
    pthread_t threads[MAX_THREAD];
    thargs_t thargs[MAX_THREAD];

    int end = min(Left.rows, MAX_THREAD);
    int step = max((int)Left.rows / end, 1);

    for (int i = 0; i < end; i++)
    {
        thargs[i].start = i*step;
        thargs[i].end = min((i+1)*step, Left.rows);
        if (pthread_create(&threads[i], NULL, multiply, (void *)&thargs[i]) != 0)
        {
            cout << "Error creating threads" << endl;
            return -1;
        }

    }

    for (int i = 0; i < end; i++)
    {
        if (pthread_join(threads[i], NULL) != 0)
        {
            cout << "Error joining threads" << endl;
            return -1;
        }
    }

    //combine and rearrange RGB channels
    RGB.col(0).copyTo(r);
    RGB.col(1).copyTo(g);
    RGB.col(2).copyTo(b);
    channels[0] = r.reshape(1, row);
    channels[1] = g.reshape(1, row);
    channels[2] = b.reshape(1, row);
    merge(channels, 3, Analy);

    cvtColor(Analy, Analy, COLOR_RGB2BGRA);
    imshow("Analygraph", Analy);
    imwrite("Analygraph.jpg", Analy);

    //compute runtime
    auto end_time = high_resolution_clock::now();
    auto running_time = duration_cast<microseconds>(end_time - start_time);
    cout << "running time: " << running_time.count()/1000 << "ms" << endl;

    waitKey(0);
    destroyAllWindows();
    
    return 0;
}
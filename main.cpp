#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "analygen.h"

#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

Mat analy_gen(Mat left, Mat right, Matrix3f M_l, Matrix3f M_r)
{
    //Mat analygraph = 0.5*left + 0.5*right;
    Mat left_channels[3];
    Mat right_channels[3];
    //left *= 1. / 255;
    split(left, left_channels);
    split(right, right_channels);

    Mat analygraph;
    //Mat analygraph = {left_channels[0], left_channels[1], left_channels[2]};
    merge(left_channels, 3, analygraph);

    // cout << (left)
    return analygraph;
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

    //regex_t regexNames;

    AnalyGen gen;

    while ((opt = getopt(argc, argv, "l:r:i:o:TGCHODR")) != -1)
    {
        switch (opt)
        {
            case 'l':
                image_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(image_path, optarg, strlen(optarg) + 1);
                cout << image_path << endl;
                left = imread(image_path);
                //left = imread(image_path, IMREAD_COLOR);
                break;
            case 'r' :
                image_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(image_path, optarg, strlen(optarg) + 1);
                cout << image_path << endl;
                right = imread(image_path, IMREAD_COLOR);
                break;
            case 'i':
                image_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(image_path, optarg, strlen(optarg) + 1);
                cout << image_path << endl;
                img = imread(image_path, IMREAD_COLOR);
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
                cout << M_l << endl;
                cout << M_r << endl;
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
                abort();
        }
    }
    //if (!left.data)
    if (!left.data || !right.data)
    {
        cout << "Error loading image" << endl;
        return 0;
    }

    out = analy_gen(left, right, M_l, M_r);

    imshow("left", left);
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
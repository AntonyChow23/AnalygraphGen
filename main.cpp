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

using namespace std;
using namespace cv;
//! [includes]
int main(int argc, char **argv)
{
    int opt;
    char *left_path = NULL;
    char *out_path = NULL;
    char *right_path = NULL;
    char *in_path = NULL;
    //regex_t regexNames;

    AnalyGen gen;

    while ((opt = getopt(argc, argv, "l:r:i:o:TGCHODR")) != -1)
    {
        switch (opt)
        {
            case 'l':
                left_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(left_path, optarg, strlen(optarg) + 1);
                cout << left_path << endl;
                gen.left = imread(left_path, IMREAD_COLOR);
                break;
            case 'r' :
                right_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(right_path, optarg, strlen(optarg) + 1);
                cout << right_path << endl;
                gen.right = imread(right_path, IMREAD_COLOR);
                break;
            case 'i':
                in_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(in_path, optarg, strlen(optarg) + 1);
                cout << in_path << endl;
                Mat img = imread(in_path, IMREAD_COLOR)
                gen.left = img(Range(0, img.size().height), Range(0, half_width));
                gen.right = img(Range(0, img.size().height), Range(half_width, img.size().width));
                gen.mix = 0.5 * left + 0.5 * right;
                break;
            case 'o':
                out_path = (char *)malloc(strlen(optarg) + 1);
                memcpy(out_path, optarg, strlen(optarg) + 1);
                break;
            case 'T':
                break;
            case 'G':
                break;
            case 'C':
                break;
            case 'H':
                break;
            case 'O':
                break;
            case 'D':
                break;
            case 'R':
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
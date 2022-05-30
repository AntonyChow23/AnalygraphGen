//#ifndef ANALYGEN_H
//#define ANALYGEN_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

class AnalyGen {
    /*
    private:
    #    char *left_path;
    #    char *right_path;
    #    char *out_path;
    #    char *in_path;
    #    Mat left;
    #    Mat right;
    #    Mat out;
    #    int function;
    */
    public:
        Mat left;
        Mat right;
        Mat out;
        int function;
        AnalyGen() {};
        AnalyGen(String in_path) {};
        AnalyGen(String left_path, String right_path){};
};
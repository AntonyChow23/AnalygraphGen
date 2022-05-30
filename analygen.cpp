#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "anaglyG.h"

#include <iostream>

using namespace std;
using namespace cv;
//! [includes]

Void::AnalyGen()
{
    left_path = NULL;
    right_path = NULL;
    in_path = NULL;
    out_path = NULL;
    left = NULL;
    right = NULL;
    out = NULL;
    function = 0;
}

Void::AnalyGen(String in_path)
{
    function = 0;
}

Void::AnalyGen(String left_path, String right_path)
{
    function = 0;
}



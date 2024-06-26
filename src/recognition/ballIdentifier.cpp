#include <cstdlib>
#include <iostream>


#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "recognition/ballIdentifier.h"
using namespace cv;

// IMPLEMENTATION OF ballType.h

BallType getBallType(Mat image, Rect2d window)
{
    Mat greyCrop;
    cvtColor(image(window),greyCrop,COLOR_BGR2GRAY);
    float fullness = ballFullness(greyCrop);
}

float ballFullness(Mat ballCrop)
{
    Mat thresholded;
    threshold(ballCrop,thresholded,0,255,THRESH_BINARY | THRESH_OTSU);

    namedWindow("W");
    imshow("W",thresholded);

    // next part should be adapted to the inscribed circle of the area
    int nonZero = countNonZero(thresholded);
    int area = ballCrop.size[0] * ballCrop.size[1];

    // return share of crop that is not black
    return (area - nonZero)/area;
}
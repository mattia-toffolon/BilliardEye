#include <cstdlib>
#include <iostream>


#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "recognition/ballIdentifier.h"
using namespace cv;

// IMPLEMENTATION OF ballType.h

/**
 * @brief Get a mask describing the approximate shape of the ball
 * 
 * @param ballCrop greyscale crop of the ball
 * @return Mat mask with same size as input image, white pixels are inscribed ellipse
 */
Mat getBallEllipse(Mat ballCrop)
{
    Mat mask = Mat::zeros(ballCrop.rows,ballCrop.cols,CV_8U);
    ellipse(
        mask,
        RotatedRect(
            Point2f(0,0),
            Point2f(mask.cols,0),
            Point2f(mask.cols,mask.rows)
        ),
        Scalar(255),
        -1
    );

    /*namedWindow("W");
    Mat scaled;
    resize(mask,scaled,Size(),10,10);
    imshow("W",scaled);
    waitKey(0);*/

    return mask;
}

BallType getBallType(Mat image, Rect2d window)
{
    // MAIN IDEA:
    // - convert relevant section of image to greyscale
    // - set the area outside the circle of the ball to black
    // - threshold the remaining area to determine how much of the ball is white
    // - return an estimation based on this percentage:
    //   - completely white: cueball
    //   - partially white: striped ball
    //   - full or almost full: solid or 8-ball depending on color
    Mat greyCrop;
    cvtColor(image(window),greyCrop,COLOR_BGR2GRAY);
    //Mat isolatedBall;
    //greyCrop.copyTo(isolatedBall,getBallEllipse(greyCrop));
    float fullness = ballFullness(greyCrop);

    return BallType::CUE;
}

float ballFullness(Mat ballCrop)
{
    //Mat equalized;
    //equalizeHist(ballCrop,equalized);
    Mat thresholded;
    threshold(ballCrop,thresholded,0,255,THRESH_BINARY | THRESH_OTSU);

    Mat isolatedBall;
    thresholded.copyTo(isolatedBall,getBallEllipse(ballCrop));

    namedWindow("W");
    Mat scaled;
    resize(ballCrop,scaled,Size(),10,10);
    imshow("W",scaled);
    waitKey(0);
    resize(isolatedBall,scaled,Size(),10,10);
    imshow("W",scaled);
    waitKey(0);

    // next part should be adapted to the inscribed circle of the area
    int nonZero = countNonZero(isolatedBall);
    int area = ballCrop.size[0] * ballCrop.size[1];

    // return share of crop that is not black
    return (area - nonZero)/area;
}
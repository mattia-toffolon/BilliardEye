#include <cstdlib>
#include <iostream>
#include <vector>


#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "recognition/ballIdentifier.h"
using namespace cv;
using std::vector;

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

// -----------------
// UTILITY FUNCTIONS
// -----------------

/**
 * @brief Utility function to get color space of image
 * 
 * @param img image to transform
 * @return std::vector<Point3f> color of pixels of img
 */
vector<Point3f> colorSpaceTransform(Mat img)
{
    vector<Point3f> ans;
    for (int r=0; r<img.rows; r++)
        for (int c=0; c<img.cols; c++)
            ans.push_back(static_cast<Point3f>(img.at<Vec3b>(r,c)));
    return ans;
}

vector<int> clusterIndexes(std::vector<Point3f> points, int k)
{
    vector<int> output;
    kmeans(
        points,
        k,
        output,
        TermCriteria(TermCriteria::Type::COUNT,100,0),
        5,
        KMEANS_PP_CENTERS
    );
    return output;
}

vector<float> clusterPercentage(Mat img, int k)
{
    vector<int> labels = clusterIndexes(colorSpaceTransform(img),k);
    vector<float> ans(k,0);
    for (int l : labels)
        ans[l]++;
    for (int i=0; i<k; i++)
    {
        ans[i] /= labels.size();
    }

    return ans;
}
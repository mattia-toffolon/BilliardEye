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

BallType getBallType(Mat ballCrop)
{
    // MAIN IDEA:
    // - convert relevant section of image to greyscale
    // - set the area outside the circle of the ball to black
    // - stretch the image brightess (localized only on the ball)
    // - threshold the remaining area to determine how much of the ball is white
    // - return an estimation based on this percentage:
    //   - completely white: cueball
    //   - partially white: striped ball
    //   - full or almost full: solid or 8-ball depending on color
    Mat greyCrop;
    cvtColor(ballCrop,greyCrop,COLOR_BGR2GRAY);
    //Mat isolatedBall;
    //greyCrop.copyTo(isolatedBall,getBallEllipse(greyCrop));
    float whiteness = ballWhiteness(greyCrop);

    if (whiteness > cueballThreshold)
        return BallType::CUE;
    else if (whiteness > stripedThreshold)
        return BallType::STRIPED;
    else
        return BallType::SOLID;
}

float ballWhiteness(Mat ballCrop,float thresh)
{
    Mat transformed = equalizedMasked(ballCrop,noArray());
    threshold(transformed,transformed,thresh*255,255,THRESH_BINARY);

    int whites = countNonZero(transformed);
    int totalPixels = transformed.rows * transformed.cols;
    return (float)whites / (float)totalPixels;
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

Mat drawClusters(Mat img, int k)
{
    Mat newImg;
    cvtColor(img,newImg,COLOR_BGR2GRAY);
    vector<int> labels = clusterIndexes(colorSpaceTransform(img),k);
    for (int r=0; r<img.rows; r++)
        for (int c=0; c<img.cols; c++)
        {
            int index = r*img.cols + c;
            newImg.at<uchar>(r,c) = (labels[index]/(k-1))*255;
        }
    return newImg;
}

Mat equalizedMasked(Mat img, InputArray mask)
{
    Mat output;

    Mat matMask;
    if (mask.empty())
    {
        matMask = getBallEllipse(img);
    }
    else
        matMask = mask.getMat();

    img.copyTo(output);
    uchar min = 255;
    uchar max = 0;
    for (int r=0; r<img.cols; r++)
    {
        for (int c=0; c<img.cols; c++)
        {
            if (matMask.at<uchar>(r,c))
            {
                min = std::min(min,img.at<uchar>(r,c));
                max = std::max(max,img.at<uchar>(r,c));
            }
        }
    }
    //std::cout << (int)min << " " << (int)max << "\n";
    for (int r=0; r<img.cols; r++)
    {
        for (int c=0; c<img.cols; c++)
        {
            if (matMask.at<uchar>(r,c))
            {
                uchar p = img.at<uchar>(r,c);
                float pp = (float)(p-min)/(float)(max-min);
                //std::cout << pp*255 << "\n";
                output.at<uchar>(r,c) = pp*255;
            }
            else
                output.at<uchar>(r,c) = 0;
        }
    }
    return output;
}
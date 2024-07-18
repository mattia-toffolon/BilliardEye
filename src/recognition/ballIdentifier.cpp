// AUTHOR: Giacomin Marco

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

BallType getBallType(Mat ballCrop, bool getCue)
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

    if (whiteness > cueballThreshold && getCue)
        return BallType::CUE;
    else if (whiteness > stripedThreshold)
        return BallType::STRIPED;
    else
        return BallType::SOLID;
}

float ballWhiteness(Mat ballCrop,float thresh)
{
    Mat transformed = equalizedMasked(ballCrop,noArray());
    Mat thresholded;
    threshold(transformed,thresholded,thresh*255,255,THRESH_BINARY);

    int whites = countNonZero(thresholded);
    int totalPixels = transformed.rows * transformed.cols;
    return (float)whites / (float)totalPixels;
}

// -----------------
// UTILITY FUNCTIONS
// -----------------

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
    for (int r=0; r<img.rows; r++)
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
    for (int r=0; r<img.rows; r++)
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

// Previous version, less robust
/*vector<Ball> _classifyBalls(Mat image, vector<Rect> windows)
{
    vector<Ball> ans;

    vector<int> cueballs;
    vector<int> striped;
    vector<int> solid;

    for (int i=0;i<windows.size();i++)
    {
        Mat window = image(windows[i]);
        switch (getBallType(window))
        {
        case BallType::CUE:
            cueballs.push_back(i);
            break;
        case BallType::STRIPED:
            striped.push_back(i);
            break;
        case BallType::SOLID:
            solid.push_back(i);
            break;
        }
    }

    // Take the brightest cueball candidate
    //assert(cueballs.size() >= 1);
    int brightest_cueball = 0;
    float max_brightness = 0;
    for (int i=0;i<cueballs.size();i++)
    {
        Mat grayCrop;
        cvtColor(image(windows[cueballs[i]]),grayCrop,COLOR_BGR2GRAY);
        float brightness = mean(grayCrop)[0]; 
        if (brightness > max_brightness)
        {
            max_brightness = brightness;
            brightest_cueball = i;
        }
    }
    Ball final_cueball = {windows[cueballs[brightest_cueball]],BallType::CUE};
    cueballs.erase(cueballs.begin() + brightest_cueball);
    ans.push_back(final_cueball);

    // Assume other cueballs are misclassified striped balls
    for (int i=0;i<cueballs.size();i++)
    {
        ans.push_back({windows[cueballs[i]],BallType::STRIPED});
    }

    // Push striped balls
    for (int i=0;i<striped.size();i++)
    {
        ans.push_back({windows[striped[i]],BallType::STRIPED});
    }

    // Take the darkest solid ball
    //assert(solid.size() >= 1);
    int darkest_solid = 0;
    float min_brightness = 255;
    for (int i=0;i<solid.size();i++)
    {
        Mat grayCrop;
        cvtColor(image(windows[solid[i]]),grayCrop,COLOR_BGR2GRAY);
        float brightness = mean(grayCrop)[0]; 
        if (brightness < min_brightness)
        {
            min_brightness = brightness;
            darkest_solid = i;
        }
    }
    Ball final_8ball = {windows[solid[darkest_solid]],BallType::EIGHT};
    solid.erase(solid.begin() + darkest_solid);
    ans.push_back(final_8ball);

    // Push rest of solid balls
    for (int i=0;i<solid.size();i++)
    {
        ans.push_back({windows[solid[i]],BallType::SOLID});
    }

    return ans;
}*/

// New version, ensures one cueball and one 8-ball are returned
/*vector<Ball> classifyBalls(Mat image, vector<Rect> windows)
{
    vector<Ball> ans;

    int brightestRec = 0;
    int darkestRec = 1;
    float maxBright = 0;
    float minBright = 255;

    for (int i=0; i<windows.size(); i++)
    {
        Mat grayCrop;
        cvtColor(image(windows[i]),grayCrop,COLOR_BGR2GRAY);

        float brightness = mean(grayCrop)[0];

        if (brightness > maxBright)
        {
            maxBright = brightness;
            brightestRec = i;
        }
        else if (brightness < minBright)
        {
            minBright = brightness;
            darkestRec = i;
        }
    }

    for (int i=0; i<windows.size(); i++)
    {
        BallType type;
        if (i == brightestRec)
            type = BallType::CUE;
        else if (i == darkestRec)
            type = BallType::EIGHT;
        else
            type = getBallType(image(windows[i]));
        
        ans.push_back({windows[i],type});
    }

    return ans;
}*/

int brightest(Mat image, vector<Rect> windows, vector<int> indexes)
{
    int ans = 0;
    float maxBright = 0;
    for (int i : indexes)
    {
        Mat grayScale;
        cvtColor(image(windows[i]),grayScale,COLOR_BGR2GRAY);
        float brightness = mean(grayScale)[0];
        if (brightness > maxBright)
        {
            ans = i;
            maxBright = brightness;
        }
    }
    return ans;
}
int brightest(Mat image, vector<Rect> windows)
{
    vector<int> indexes;
    for (int i=0;i<windows.size();i++)
        indexes.push_back(i);
    return brightest(image,windows,indexes);
}

int darkest(Mat image, vector<Rect> windows, vector<int> indexes)
{
    int ans = 0;
    float minBright = 255;
    for (int i : indexes)
    {
        Mat grayScale;
        cvtColor(image(windows[i]),grayScale,COLOR_BGR2GRAY);
        float brightness = mean(grayScale)[0];
        if (brightness < minBright)
        {
            ans = i;
            minBright = brightness;
        }
    }
    return ans;
}
int darkest(Mat image, vector<Rect> windows)
{
    vector<int> indexes;
    for (int i=0;i<windows.size();i++)
        indexes.push_back(i);
    return darkest(image,windows,indexes);
}

vector<Ball> classifyBalls(Mat image, vector<Rect> windows)
{
    vector<int> cueballs;
    vector<int> striped;
    vector<int> solid;

    for (int i=0;i<windows.size();i++)
    {
        Mat window = image(windows[i]);
        switch (getBallType(window,true))
        {
        case BallType::CUE:
            cueballs.push_back(i);
            break;
        case BallType::STRIPED:
            striped.push_back(i);
            break;
        case BallType::SOLID:
            solid.push_back(i);
            break;
        }
    }

    int cueball;
    if (cueballs.size() > 0)
    {
        // Get brightest cueball
        cueball = brightest(image,windows,cueballs);
    }
    else
    {
        // Get brightest ball in general
        cueball = brightest(image,windows);
    }

    int eightball;
    if (solid.size() > 0)
    {
        // Get darkest solid ball
        eightball = darkest(image,windows,solid);
    }
    else
    {
        // Get darkest ball in general
        eightball = darkest(image,windows);
    }

    vector<Ball> ans;
    for (int i = 0; i<windows.size(); i++)
        if (i != cueball && i != eightball)
            ans.push_back({windows[i],getBallType(image(windows[i]))});
    ans.push_back({windows[cueball],BallType::CUE});
    ans.push_back({windows[eightball],BallType::EIGHT});
    return ans;
}

Mat magnifyImg(Mat img, float magnification)
{
    Mat ans;
    resize(img,ans,Size(0,0),magnification,magnification);
    return ans;
}
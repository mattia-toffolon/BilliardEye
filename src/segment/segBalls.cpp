#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "segment/segBalls.h"
#include "utils/getTableColor.hpp"

using namespace cv;
using namespace std;

// UNUSABLE
// Mat meanShiftClustering(Mat& inputImage, int spatialRadius, int colorRadius, int maxLevel)
// {
//     Mat labImage;
//     cvtColor(inputImage, labImage, COLOR_BGR2Lab);

//     TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, maxLevel, 1.0);
//     Size windowSize(4 * spatialRadius + 1, 4 * spatialRadius + 1);

//     Mat outputImage;
//     pyrMeanShiftFiltering(labImage, outputImage, spatialRadius, colorRadius, maxLevel, criteria);

//     cvtColor(outputImage, outputImage, COLOR_Lab2BGR);

//     return outputImage;
// }

vector<Vec3f> circlesFinder(Mat img, int method, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius, bool draw) {
    std::vector<cv::Vec3f> out;
    HoughCircles(img, out, method, dp, minDist, param1, param2, minRadius, maxRadius);

    if(out.empty()) cout<<"No circles found"<<endl;

    if(draw) {
        for(Vec3i c : out) {
            Point center = Point(c[0], c[1]);
            int radius = c[2];
            circle(img, center, radius, Scalar(0,0,255), 2, LINE_AA);
        }
        imshow("window", img);
        waitKey(0);
    }

    return out;
}

vector<Rect> bboxConverter(vector<Vec3f> circles) {
    vector<Rect> bboxes;
    for(Vec3i c : circles) bboxes.push_back(Rect(c[0]-c[2], c[1]-c[2], 2*c[2], 2*c[2]));
    
    return bboxes;
}

float squaredEuclideanDist(Vec3b pixel, Vec3b center) {
    float sum = 0;
    for(int i=0; i<3; i++)
        sum += (pixel[i]-center[i]) * (pixel[i]-center[i]);
    return sum;
}


Vec3b getClusterCentroid(Vec3b pixel, vector<Vec3b> centers) {
    Vec3b ret = centers[0];
    float minDist = squaredEuclideanDist(pixel, centers[0]);
    for(int i=1; i<centers.size(); i++) {
        float dist = squaredEuclideanDist(pixel, centers[i]);
        if(dist < minDist) {
            minDist = dist;
            ret = centers[i];
        }
    }
    return ret;
}

// To be perfected...
Mat quantizeColors(Mat img) {

    vector<Vec3b> colors;
    colors.push_back(Vec3b(255, 255, 255));   // 0  - White 
    colors.push_back(Vec3b(0, 178, 178));     // 1  - Yellow
    colors.push_back(Vec3b(178, 0, 0));       // 2  - Blue (dark)
    colors.push_back(Vec3b(230, 0, 0));       // 2  - Blue (light)
    colors.push_back(Vec3b(0, 0, 178));       // 3  - Red 
    colors.push_back(Vec3b(90, 0, 90));       // 4  - Purple 
    colors.push_back(Vec3b(0, 115, 178));     // 5  - Orange
    colors.push_back(Vec3b(0, 178, 0));       // 6  - Green
    colors.push_back(Vec3b(0, 0, 90));        // 7  - Maroon
    colors.push_back(Vec3b(0, 0, 0));         // 8  - Black 
    colors.push_back(Vec3b(180, 180, 180));   // 9  - Gray (light)
    colors.push_back(Vec3b(80, 80, 80));      // 10 - Gray (dark)    
    colors.push_back(getTableColor(img));     // 11 - Table color

    Mat out = Mat(img);

    for(int i=0; i<out.rows; i++) {
        for(int j=0; j<out.cols; j++) {
            if(out.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) continue;
            out.at<Vec3b>(i, j) = getClusterCentroid(out.at<Vec3b>(i, j), colors);
        }
    }

    return out;
}
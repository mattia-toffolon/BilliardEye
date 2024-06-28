#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "segment/segBalls.h"

using namespace cv;
using namespace std;

// UNUSABLE
Mat meanShiftClustering(Mat& inputImage, int spatialRadius, int colorRadius, int maxLevel)
{
    Mat labImage;
    cvtColor(inputImage, labImage, COLOR_BGR2Lab);

    TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, maxLevel, 1.0);
    Size windowSize(4 * spatialRadius + 1, 4 * spatialRadius + 1);

    Mat outputImage;
    pyrMeanShiftFiltering(labImage, outputImage, spatialRadius, colorRadius, maxLevel, criteria);

    cvtColor(outputImage, outputImage, COLOR_Lab2BGR);

    return outputImage;
}

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


// cv::Mat quantizeColors(cv::Mat& image, int k) {
//     return;
// }
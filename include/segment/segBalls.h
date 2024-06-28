#ifndef SEG_BALLS
#define SEG_BALLS

#include <opencv2/highgui.hpp>

cv::Mat meanShiftClustering(cv::Mat& inputImage, int spatialRadius=10, int colorRadius=20, int maxLevel=1);

std::vector<cv::Vec3f> circlesFinder(cv::Mat img, int method, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius, bool draw=true);

std::vector<cv::Rect> bboxConverter(std::vector<cv::Vec3f> circles);

// cv::Mat quantizeColors(cv::Mat& image, char tableColor[]);

#endif
#ifndef SEG_BALLS
#define SEG_BALLS

#include <opencv2/highgui.hpp>

// cv::Mat meanShiftClustering(cv::Mat& inputImage, int spatialRadius=10, int colorRadius=20, int maxLevel=1);

std::vector<cv::Vec3f> circlesFinder(cv::Mat img, int method, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius, bool draw=true);

std::vector<cv::Rect> bboxConverter(std::vector<cv::Vec3f> circles);

std::vector<cv::Vec3f> circlesFilter(cv::Mat img, std::vector<cv::Vec3f> circles, std::vector<cv::Vec3b> tableColors, int levels, bool draw);

std::vector<cv::Vec3f> circlesFilter2(cv::Mat img, std::vector<cv::Vec3f> circles, bool draw);

std::vector<cv::Vec3f> smartCircleMerge(std::vector<cv::Vec3f> first, std::vector<cv::Vec3f> second);

float squaredEuclideanDist(cv::Vec3b pixel, cv::Vec3b center);

cv::Vec3b getClusterCentroid(cv::Vec3b pixel, std::vector<cv::Vec3b> centers); 

// cv::Mat quantizeColors(cv::Mat img, int delta);

cv::Mat subtractTable(cv::Mat img);

std::vector<cv::Rect> getBBoxes(cv::Mat img, cv::Mat last, cv::Mat tableMask, std::vector<cv::Point2f> points);

#endif
#ifndef SEG_BALLS
#define SEG_BALLS

#include <opencv2/highgui.hpp>


std::vector<cv::Vec3f> circlesFinder(cv::Mat img, int method, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius, bool draw=true);

std::vector<cv::Rect> bboxConverter(std::vector<cv::Vec3f> circles);

std::vector<cv::Vec3f> circlesFilter(cv::Mat img, std::vector<cv::Vec3f> circles, std::vector<cv::Vec3b> tableColors, int levels, bool draw);

std::vector<cv::Vec3f> refineCircles(cv::Mat img, std::vector<cv::Rect> bboxes, bool draw);

std::vector<cv::Vec3f> smartCircleMerge(std::vector<cv::Vec3f> first, std::vector<cv::Vec3f> second);

float squaredEuclideanDist(cv::Vec3b pixel, cv::Vec3b center);

cv::Mat subtractTable(cv::Mat img);

// DATEMI IL PRIMO FRAME PRESO DA "FRAMES"
std::vector<cv::Rect> getBBoxes(cv::Mat img, cv::Mat mask, cv::Mat transf);

#endif
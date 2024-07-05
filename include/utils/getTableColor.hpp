#ifndef GET_TABLE_COLOR
#define GET_TABLE_COLOR

#include <opencv2/highgui.hpp>

cv::Vec3b getTableColor(cv::Mat& img);

cv::Vec3b getUpperTableColor(cv::Mat& img);

cv::Vec3b getLowerTableColor(cv::Mat& img);

std::vector<cv::Vec3b> getTableColorVariations(cv::Mat& img, bool HSV);

#endif
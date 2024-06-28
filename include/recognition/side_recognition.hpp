#ifndef SIDE_RECOGNITION
#define SIDE_RECOGNITION
#include <opencv2/imgproc.hpp>
std::vector<cv::Mat> getRotatedborders(const std::vector<cv::Point2f> points, const cv::Mat img, int thick = 20);
bool isShortFirst(std::vector<cv::Mat> sides);
#endif

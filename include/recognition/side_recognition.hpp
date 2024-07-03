#ifndef SIDE_RECOGNITION
#define SIDE_RECOGNITION
#include <opencv2/imgproc.hpp>
//returns the rectangle along the sides of the table
//the points should be ordered (see segment/segTable.h->orderPoints)
//thick: the width of the rectangle orthogonal to the side
std::vector<cv::Mat> getRotatedborders(const std::vector<cv::Point2f> points, const cv::Mat img, int thick = 20);
//returns true if the (ordered) sides have a short side as first
bool isShortFirst(std::vector<cv::Mat> sides);
#endif

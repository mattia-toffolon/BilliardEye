#ifndef BBOXES_UTILS
#define BBOXES_UTILS

#include <opencv2/highgui.hpp>

cv::Rect toRect(cv::Vec3f circle);

cv::Vec3f toCircle(cv::Rect box);

void drawBBoxes(cv::Mat img, std::vector<cv::Rect> bboxes);

void drawBBoxesCanvas(cv::Mat img, std::vector<cv::Rect> bboxes1, std::vector<cv::Rect> bboxes2);

std::vector<cv::Rect> expandBBoxes(std::vector<cv::Rect> bboxes, const float MULT);

#endif
#ifndef DRAW_BBOXES
#define DRAW_BBOXES

#include <opencv2/highgui.hpp>

void drawBBoxes(cv::Mat img, std::vector<cv::Rect> bboxes);

void drawBBoxesCanvas(cv::Mat img, std::vector<cv::Rect> bboxes1, std::vector<cv::Rect> bboxes2);

std::vector<cv::Rect> expandBBoxes(std::vector<cv::Rect> bboxes, const float MULT);

#endif
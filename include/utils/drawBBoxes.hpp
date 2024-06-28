#ifndef DRAW_BBOXES
#define DRAW_BBOXES

#include <opencv2/highgui.hpp>

void drawBBoxes(cv::Mat img, std::vector<cv::Rect> bboxes);

#endif
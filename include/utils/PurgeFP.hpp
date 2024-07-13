#ifndef PURGEFP
#define PURGEFP
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

std::vector<cv::Rect> purgeFP(cv::Mat img, cv::Mat persp_trans, std::vector<cv::Rect> bboxes);

std::vector<cv::Rect> purgeByCanny(cv::Mat img, cv::Mat persp_trans, std::vector<cv::Rect> bboxes, std::vector<float>& prev_vals);


#endif

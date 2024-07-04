#ifndef PERFTESTING_H
#define PERFTESTING_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/balls.hpp"

/**
 * @brief Calculates IoU (percentage) of two `Rect`s
 * 
 * @param region1 first rectangle to compare
 * @param region2 second rectangle to compare
 * @return float IoU
 */
float intersectionOverUnion(cv::Rect region1, cv::Rect region2);

#endif /* PERFTESTING_H */

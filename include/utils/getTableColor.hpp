// AUTHOR: Toffolon Mattia

#ifndef GET_TABLE_COLOR
#define GET_TABLE_COLOR

#include <opencv2/highgui.hpp>

/**
 * Function that estimates the table cloth color
 * 
 * @param img image on which the table cloth color shall estimated
 * @return estimated table cloth color shall estimated
 */
cv::Vec3b getTableColor(cv::Mat img);

#endif
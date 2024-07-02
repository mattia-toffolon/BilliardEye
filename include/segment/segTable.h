#ifndef SEG_TABLE
#define SEG_TABLE
#include <opencv2/core/matx.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
//simple struct used to represent lines
struct linestr{
    cv::Point2f start;
    cv::Point2f stop;
};

//tells whether the lines are similar within a threshold
bool arelinessimilar(const struct linestr a, const struct linestr b, double thresh);

//return a binary mask of the most centered of the k clusters
//returned by applying kmeans to image in
//blurSize: GaussianBlur to apply to image in as preprocessing
cv::Mat nonbinarykmeans(const cv::Mat in, int k=3, int blurSize=31);

//returns the greatest connected component in the binary image input
//in order to avoid weakly connected components a OPEN morphological
//operation is first applied
cv::Mat greatest_island(cv::Mat input);
//returns the first 4 distinct enough (see arelinessimilar) lines
//contained in the image img
std::vector<struct linestr> line4line(cv::Mat img, double thresh);

//given four lines it returns the 4 vertices with x in [0, max_col] and 
//y in [0,max_row]
std::vector<cv::Point2f> find_vertices(std::vector<struct linestr> lines, int max_col, int max_row);

//orders the points clockwise, starting from the leftmost of the two
//uppermost points
std::vector<cv::Point2f> order_points(std::vector<cv::Point2f> point4);

//returns the mean color of img withing mask
cv::Vec3b meanMask(cv::Mat img, cv::Mat mask);

//applies k-means to in, using "colors" as final result to 
//color the output image
cv::Mat simplekmeans(const cv::Mat in, int k, char* colors);

//returns a binary mask of in where only the pixels with 
//hue within hue(color)+-thresh have value different from 0
cv::Mat threshHue(const cv::Mat in, const cv::Vec3b color, int thresh=5);

std::vector<cv::Point2f> find_table(cv::Mat img, cv::Mat &mask);
//segmentation pipeline:
//1)nonbinarykmeans with k=3 (2 leads to undersementation, 4 to oversegmentation)
//2)findgreatest island
//3)from given mask obtain mean color
//4)threshhold by hue with threshol=5 (empirically very robust)
//5)find lines with obtained mask
//6)find vertices from given lines
//7)profit
#endif

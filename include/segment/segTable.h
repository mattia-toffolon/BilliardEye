#ifndef SEG_TABLE
#define SEG_TABLE
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

struct linestr{
    cv::Point2f start;
    cv::Point2f stop;
};
bool arelinessimilar(const struct linestr a, const struct linestr b, double thresh);
cv::Mat nonbinarykmeans(const cv::Mat in, int k=3, int blurSize=31);
cv::Mat greatest_island(cv::Mat input);
std::vector<struct linestr> line4line(cv::Mat img, double thresh);
std::vector<cv::Point2f> find_vertices(std::vector<struct linestr> lines, int max_col, int max_row);
std::vector<cv::Point> order_points(std::vector<cv::Point> point4);
cv::Vec3b meanMask(cv::Mat img, cv::Mat mask);
cv::Mat simplekmeans(const cv::Mat in, int k, char* colors);
#endif

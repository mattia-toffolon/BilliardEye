#ifndef SEG_TABLE
#define SEG_TABLE
#include <opencv2/core/matx.hpp>
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
cv::Mat threshHue(const cv::Mat in, const cv::Vec3b color, int thresh=5);
//segmentation pipeline right now:
//1)nonbinarykmeans with k=3 (2 leads to undersementation, 4 to oversegmentation)
//2)findgreatest island
//3)from given mask obtain mean color
//4)threshhold by hue with threshol=5 (empirically very robust)
//5)find lines with obtained mask
//6)find vertices from given lines
//7)profit
#endif

#ifndef BALLS
#define BALLS
#include <opencv2/core.hpp>
enum BallType{
    CUE = 1,
    EIGHT = 2,
    SOLID = 3,
    STRIPED = 4
};
struct Ball{
    cv::Rect2d bbox;
    BallType type; 
};
std::vector<Ball> readBallsFile(std::string filename);
#endif

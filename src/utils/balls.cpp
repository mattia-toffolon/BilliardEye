#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <fstream>
#include "utils/balls.hpp"

std::vector<Ball> readBallsFile(std::string filename){
    std::ifstream gt(filename);
    int x, y, width,height, tip;
    std::vector<Ball> balls;
    while(gt >> x){
        gt>>y>>width>>height>>tip;
        balls.push_back(Ball{cv::Rect2d(cv::Point2d(x,y),cv::Point2d(x+width,y+height)),static_cast<BallType>(tip)});
    }
    return balls;
}

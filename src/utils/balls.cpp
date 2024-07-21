// AUTHOR: Artico Giovanni

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <fstream>
#include <iostream>
#include "utils/balls.hpp"

std::vector<Ball> readBallsFile(std::string filename){
    std::ifstream gt(filename);
    if(!gt.is_open()){
        std::cout << "file " + filename + " not found\n";
        throw gt;
    }
    int x, y, width,height, tip;
    std::vector<Ball> balls;
    while(gt >> x){
        gt>>y>>width>>height>>tip;
        balls.push_back(Ball{cv::Rect2d(cv::Point2d(x,y),cv::Point2d(x+width,y+height)),static_cast<BallType>(tip)});
    }
    gt.close();
    return balls;
}

void writeBallsFile(std::string filename, std::vector<Ball> balls){
    std::ofstream file(filename);
    if(!file.is_open()){
        std::cout << "file " + filename + "not found\n";
        throw file;
    }
    for(int i = 0; i < balls.size(); i++){
        auto b = balls[i];
        file << b.bbox.x << " " << b.bbox.y << " " << b.bbox.width << " "<< b.bbox.height << " " << b.type;
        if(i != balls.size() - 1){
            file << std::endl;
        }
    }
    file.close();
}

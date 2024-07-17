//Giovanni Artico
#include "recognition/transformPoints.hpp"

using namespace cv;

Mat transPoints(std::vector<Point2f> points, int width, int height, bool rotated){
    std::vector<Point2f> dst;
    if(!rotated){
        dst = std::vector<Point2f>{
                Point2f(width,0),
                Point2f(width, height),
                Point2f(0, height),
                Point2f(0,0)
        };
    }
    else{
        dst = std::vector<Point2f>{
                Point2f(0,0),
                Point2f(width,0),
                Point2f(width, height),
                Point2f(0, height)
        };
    }
    return getPerspectiveTransform(points, dst);
}

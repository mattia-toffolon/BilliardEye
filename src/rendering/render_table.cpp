#include "rendering/render_table.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/drawBBoxes.hpp"
#include <algorithm>
using namespace cv;

Scalar getColor(BallType b){
    if(b == BallType::CUE){
        return Scalar(255, 255, 255);
    }
    else if(b == BallType::EIGHT){
        return Scalar(0, 0, 0);
    }
    else if(b == BallType::SOLID){
        return Scalar(255, 0, 0);
    }
    else{
        return Scalar(0, 0, 255);
    }
}

TableRenderer::TableRenderer(VideoReader v, TrackBalls t, std::vector<Ball> balls, cv::Mat transform, int width, int cols) : tracker(t), vid(v), transform(transform.clone()){
    this->curimg = Mat::zeros(Size(width, cols), CV_8UC3);
    curimg.setTo(Scalar(255, 255,255));
    for(auto r : balls){
        Ball curb{r.bbox, r.type};
        bbs.push_back(curb);
    }
}

cv::Mat TableRenderer::nextFrame(){
    Mat fram = this->vid.nextFrame();
    if(fram.rows == 0){
        return fram;
    }
    std::vector<int> removed;
    const std::vector<Ball> newballs = tracker.update(fram, removed);
    std::vector<Rect> bounding;
    for(auto b : newballs){
        bounding.push_back(b.bbox);
    }
    drawBBoxes(fram, bounding);
    int real = 0;
    for(int i = 0; i < bbs.size(); i++){
        if(std::find(removed.begin(), removed.end(), i) != removed.end()){
            continue;
        }    
        Point2f oldcenter = Point2f(bbs[i].bbox.x+bbs[i].bbox.width/2,bbs[i].bbox.y+bbs[i].bbox.height/2);
        Point2f newCenter = Point2f(newballs[real].bbox.x+newballs[real].bbox.width/2,newballs[real].bbox.y+newballs[real].bbox.height/2);
        std::vector<Point2f> old{oldcenter};
        std::vector<Point2f> niu{newCenter};
        perspectiveTransform(old, old, transform);
        perspectiveTransform(niu, niu, transform);
        line(curimg, old[0], niu[0], Scalar(0,0,0));
        bbs[i].bbox = newballs[real].bbox;
        real++;
    }
    for(int i = removed.size()-1; i >=0; i--){
        bbs.erase(bbs.begin()+removed[i],bbs.begin()+removed[i]+1);
    }
    Mat screen = curimg.clone();
    for(Ball b : bbs){
        if(b.bbox.x < 0){
            continue;
        }
        Point2f c = Point2f(b.bbox.x+b.bbox.width/2,b.bbox.y+b.bbox.height/2);
        std::vector<Point2f> vec{c};
        perspectiveTransform(vec, vec, transform);
        circle(screen, vec[0], 10, getColor(b.type), FILLED);
        circle(screen, vec[0], 11, Scalar(0,0,0), 2, LINE_AA);
    }
    return screen;
}

std::vector<Ball> TableRenderer::getBalls(){
    return this->bbs;
}

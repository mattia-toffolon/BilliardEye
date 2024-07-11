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

TableRenderer::TableRenderer(VideoReader v, TrackBalls t, std::vector<Ball> balls, cv::Mat transform, int width, int height) : tracker(t), vid(v), transform(transform.clone()){
    this->curimg = Mat::zeros(Size(width, height), CV_8UC3);
    curimg.setTo(Scalar(255, 255,255));
    for(auto r : balls){
        Ball curb{r.bbox, r.type};
        bbs.push_back(curb);
    }
    holes.push_back(Point(0, 0));
    holes.push_back(Point(width/2, 0));
    holes.push_back(Point(width, 0));
    holes.push_back(Point(width, height));
    holes.push_back(Point(width/2, height));
    holes.push_back(Point(0, height));
}

bool TableRenderer::is_holed(Point ball){
    for(auto p : holes){
        if(norm(p-ball) < hole_radius){
            return true;
        }
    }
    return false;
}
cv::Mat TableRenderer::nextFrame(){
    Mat fram = this->vid.nextFrame();
    if(fram.rows == 0){
        return fram;
    }
    std::vector<int> removed,keep;
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
        if(old[0] != niu[0] && is_holed(niu[0])){
            std::cout << "YIPPEEEE\n";
            removed.push_back(i);
        }
        else{
            keep.push_back(real);
        }
        bbs[i].bbox = newballs[real].bbox;
        real++;
    }
    this->tracker.removeBalls(keep, fram);
    std::sort(removed.begin(), removed.end());
    for(int i = removed.size()-1; i >=0; i--){
        bbs.erase(bbs.begin()+removed[i],bbs.begin()+removed[i]+1);
    }

    Mat screen = curimg.clone();
    const int BALL_RAD_RATIO = 70; 
    for(Ball b : bbs) {
        Point2f c = Point2f(b.bbox.x+b.bbox.width/2,b.bbox.y+b.bbox.height/2);
        std::vector<Point2f> vec{c};
        perspectiveTransform(vec, vec, transform);
        int rad = curimg.cols / BALL_RAD_RATIO;
        circle(screen, vec[0], rad, getColor(b.type), FILLED);
        circle(screen, vec[0], rad, Scalar(0,0,0), 1, LINE_AA);
    }

    const std::string TABLE_PATH = "../data/table.png";
    Mat table = imread(TABLE_PATH, IMREAD_UNCHANGED);

    const float TABLE_RATIO = 1.3;
    resize(table, table, Size(screen.cols*TABLE_RATIO, screen.rows*TABLE_RATIO));

    Mat black = Mat::zeros(table.size(), CV_8UC3);
    Mat white = Mat(table.size(), CV_8UC3, Scalar(255, 255, 255));

    std::vector<Mat> table_channels(4);
    split(table, table_channels);
    Mat mask = table_channels[3];

    int delta_x = (table.cols - screen.cols) / 2;
    int delta_y = (table.rows - screen.rows) / 2;   
    Rect roi = Rect(delta_x, delta_y, screen.cols, screen.rows);
    screen.copyTo(white(roi));

    black.copyTo(white, mask);

    return white;
}

std::vector<Ball> TableRenderer::getBalls(){
    return this->bbs;
}

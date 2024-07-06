#include "tracking/TrackBalls.hpp"
#include "opencv2/core/types.hpp"
using namespace cv;

TrackBalls::TrackBalls(Mat frame, std::vector<Ball> bb){
    legacy::TrackerCSRT::Params params;
    for(auto r : bb){
        Ball curb{r.bbox, r.type};
        bbs.push_back(curb);
        auto cur = TrackerCSRT::create();
        cur->init(frame, r.bbox);
        multi.push_back(cur);
    }
}

std::vector<Ball> TrackBalls::update(Mat frame){
    for(int i = 0; i <multi.size(); i++){
        if(bbs[i].bbox.x < 0){
            continue;
        }
        auto tr = multi[i];
        Rect bb = bbs[i].bbox;
        bool tracked = tr->update(frame, bb);
        if(tracked){
            bbs[i].bbox = bb;
        }
        else{
            bbs[i].bbox = Rect(-1,-1, 0, 0);
        }
    }
    return bbs;
}

#include "tracking/TrackBalls.hpp"
using namespace cv;

TrackBalls::TrackBalls(Mat frame, std::vector<Ball> bb){
    legacy::TrackerCSRT::Params params;
    for(auto r : bb){
        bbs.push_back(r);
        auto cur = TrackerCSRT::create();
        cur->init(frame, r.bbox);
        multi.push_back(cur);
    }
}

std::vector<Ball> TrackBalls::update(Mat frame){
    std::vector<int> removed;
    for(int i = 0; i <multi.size(); i++){
        auto tr = multi[i];
        Rect bb = Rect(bbs[i].bbox);
        bool tracked = tr->update(frame, bb);
        if(tracked){
            bbs[i].bbox = bb;
        }
        else{
            removed.push_back(i);
        }
    }
    for(int i = removed.size()-1; i >=0; i--){
        multi.erase(multi.begin()+removed[i],multi.begin()+removed[i]+1);
        bbs.erase(bbs.begin()+removed[i],bbs.begin()+removed[i]+1);
    }
    return bbs;
}

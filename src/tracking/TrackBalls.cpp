#include "tracking/TrackBalls.hpp"
#include "opencv2/core/types.hpp"
#include "segment/segBalls.h"
#include "segment/segTable.h"
#include "recognition/side_recognition.hpp"
#include "utils/drawBBoxes.hpp"
#include <set>

using namespace cv;
using namespace std;

TrackBalls::TrackBalls(Mat frame, vector<Ball> bb){
    legacy::TrackerCSRT::Params params;
    for(auto r : bb) {
        Ball curb{r.bbox, r.type};
        bbs.push_back(curb);
        auto cur = TrackerCSRT::create();
        cur->init(frame, r.bbox);
        multi.push_back(cur);
    }
}

vector<Ball> TrackBalls::update(Mat frame){
    vector<int> lost_indexes, found_indexes;

    for(int i = 0; i < multi.size(); i++) {
        // if(bbs[i].bbox.x < 0){
        //     continue;
        // }
        auto tr = multi[i];
        Rect bb = bbs[i].bbox;
        bool tracked = tr->update(frame, bb);
        if(tracked) {
            bbs[i].bbox = bb;
            found_indexes.push_back(i);
        }
        else {
            // bbs[i].bbox = Rect(-1,-1, 0, 0);
            lost_indexes.push_back(i);
        }
    }

    if(lost_indexes.size() > 0) {
        cout<<"Lost balls: ";
        for(int i : lost_indexes) cout<<i<<" ";
        cout<<endl;

        Mat mask;
        vector<Point2f> points = find_table(frame, mask);
        Mat transf = getTransformation(frame, points);

        vector<Rect> found_bboxes = getBBoxes(frame, mask, transf);
        // drawBBoxes(frame, found_bboxes);

        adjustBalls(bbs, found_indexes, lost_indexes, found_bboxes, frame);

        for(int i : lost_indexes) multi[i]->init(frame, bbs[i].bbox);
    }

    return bbs;
}

float TrackBalls::sqEuclideanDist(Rect r1, Rect r2) {
    float dist = (r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y);
    return dist; 
}


int TrackBalls::getClosestBBoxIndex(Rect tracked, vector<Rect> found) {
    if(found.empty()) return -1;

    float min_dist = sqEuclideanDist(tracked, found[0]);
    int index = 0;
    for(int i=1; i<found.size(); i++) {
        float dist = sqEuclideanDist(tracked, found[i]);
        index = (dist < min_dist ? i : index);
        min_dist = (dist < min_dist ? dist : min_dist);
    }

    return index;
}

vector<Ball> TrackBalls::adjustBalls(vector<Ball>&  balls, vector<int> found_indexes, vector<int> lost_indexes, vector<Rect> found_bboxes, Mat frame) {
    
    // drawBBoxes(frame, found_bboxes);

    for(int i : found_indexes) {
        int relative_bbox_index = getClosestBBoxIndex(balls[i].bbox, found_bboxes);
        // cout<<balls[i].bbox<<" --- "<<found_bboxes[relative_bbox_index]<<endl;
        found_bboxes.erase(found_bboxes.begin() + relative_bbox_index);
    }

    // drawBBoxes(frame, found_bboxes);

    for(int i : lost_indexes) {
        int relative_bbox_index = getClosestBBoxIndex(balls[i].bbox, found_bboxes);
        balls[i].bbox = found_bboxes[relative_bbox_index];
    }

    return balls;
}
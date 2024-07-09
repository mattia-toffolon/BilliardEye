#include "tracking/TrackBalls.hpp"
#include "opencv2/core/types.hpp"
#include "segment/segBalls.h"
#include "segment/segTable.h"
#include "recognition/side_recognition.hpp"
#include "utils/drawBBoxes.hpp"
#include "utils/perfTesting.h"
#include <set>

using namespace cv;
using namespace std;

TrackBalls::TrackBalls(Mat frame, vector<Ball> bb) {
    // legacy::TrackerCSRT::Params params;
    const int delta = 5;

    for(auto ball : bb) {
        Rect aug_bbox = Rect(ball.bbox.x-delta, ball.bbox.y-delta, ball.bbox.width+2*delta, ball.bbox.height+2*delta);
        Ball aug_ball{aug_bbox, ball.type};
        balls.push_back(aug_ball);
        auto cur = TrackerCSRT::create();
        cur->init(frame, aug_bbox);
        multi_tracker.push_back(cur);
    }
}

vector<Ball> TrackBalls::update(Mat frame, vector<int>& renderer_remove_idxs){
    const float iou_thr = 0.8;
    vector<int> lost_indexes, found_indexes;

    for(int i = 0; i < multi_tracker.size(); i++) {
        auto tr = multi_tracker[i];
        Rect bb = balls[i].bbox;
        bool tracked = tr->update(frame, bb);
        if(tracked) {
            if(intersectionOverUnion(balls[i].bbox, bb) < iou_thr) balls[i].bbox = bb;
            found_indexes.push_back(i);
        }
        else {
            lost_indexes.push_back(i);
        }
    }

    if(lost_indexes.size() > 0) {
        // cout<<"Lost balls: ";
        // for(int i : lost_indexes) cout<<i<<" ";
        // cout<<endl;

        Mat mask;
        vector<Point2f> points = find_table(frame, mask);
        Mat transf = getTransformation(frame, points);

        vector<Rect> found_bboxes = getBBoxes(frame, mask, transf);
        // drawBBoxes(frame, found_bboxes);

        adjustBalls(found_indexes, lost_indexes, found_bboxes, frame, renderer_remove_idxs);
    }

    return balls;
}

void TrackBalls::removeBalls(vector<int> indexKeepList, Mat frame) {
    vector<Ball> new_balls;
    vector<Ptr<TrackerCSRT>> new_multi_tracker;
    for(int i : indexKeepList) {
        new_balls.push_back(balls[i]);
        auto tracker = TrackerCSRT::create();
        tracker->init(frame, balls[i].bbox);
        new_multi_tracker.push_back(tracker);
    }

    balls = new_balls;
    multi_tracker = new_multi_tracker;

    return;
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

void TrackBalls::adjustBalls(vector<int> found_indexes, vector<int> lost_indexes, vector<Rect> found_bboxes, Mat frame, vector<int>& renderer_remove_idxs) {
    
    if(balls.size() > found_bboxes.size()) {
        // cout<<"NOT ENOUGH BALLS"<<endl;
        removeBalls(found_indexes, frame);
        renderer_remove_idxs = lost_indexes;
        return;
    }

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

    for(int i : lost_indexes) multi_tracker[i]->init(frame, balls[i].bbox);

    return;
}
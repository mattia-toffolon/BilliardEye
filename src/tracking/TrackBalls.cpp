// AUTHOR: Toffolon Mattia

#include "tracking/TrackBalls.hpp"
#include "opencv2/core/types.hpp"
#include "segment/segBalls.h"
#include "segment/segTable.h"
#include "recognition/side_recognition.hpp"
#include "utils/bboxesUtils.hpp"
#include "utils/perfTesting.h"
#include <set>

using namespace cv;
using namespace std;

TrackBalls::TrackBalls(Mat frame, vector<Ball> bb) {
    // Each ball has its box expanded and the relative tracker initialized
    for(auto ball : bb) {
        Rect aug_bbox = Rect(ball.bbox.x-DELTA, ball.bbox.y-DELTA, ball.bbox.width+2*DELTA, ball.bbox.height+2*DELTA);
        Ball aug_ball{aug_bbox, ball.type};
        balls.push_back(aug_ball);
        auto cur = TrackerCSRT::create();
        cur->init(frame, aug_bbox);
        multi_tracker.push_back(cur);
    }
}

vector<Ball> TrackBalls::update(Mat frame, vector<int>& renderer_remove_idxs) {
    const float IOU_THR = 0.8;
    vector<int> lost_indexes, found_indexes;

    // Each tracker is updated with the next video frame. Indexes of correctly and wrongly updated balls are kept in distinct vectors
    for(int i = 0; i < multi_tracker.size(); i++) {
        auto tr = multi_tracker[i];
        Rect bb = balls[i].bbox;
        bool tracked = tr->update(frame, bb);
        // A ball is updated if IoU between new and old box is below IOU_THR threshold
        if(tracked) {
            if(intersectionOverUnion(balls[i].bbox, bb) < IOU_THR) balls[i].bbox = bb;
            found_indexes.push_back(i);
        }
        else {
            lost_indexes.push_back(i);
        }
    }

    // If some balls are lost, ball detection algorithm is called along with adjustBalls that handles the problem
    if(lost_indexes.size() > 0) {
        Mat mask;
        vector<Point2f> points = find_table(frame, mask);
        Mat transf = getTransformation(frame, points);

        vector<Rect> found_bboxes = getBBoxes(frame, mask, transf);

        adjustBalls(found_indexes, lost_indexes, found_bboxes, frame, renderer_remove_idxs);
    }

    return balls;
}

void TrackBalls::removeBalls(vector<int> indexEraseList) {
    // Balls and trackers which index is the given vector are erased
    for(int i=indexEraseList.size()-1; i>=0; i--) {
        balls.erase(balls.begin() + indexEraseList[i]);
        multi_tracker.erase(multi_tracker.begin() + indexEraseList[i]);
    }

    return;
}

vector<Ball> TrackBalls::getRealBalls() {
    // Balls with original box width are computed and returned
    vector<Ball> ret;
    for(Ball ball : balls) {
        Rect red_bbox = Rect(ball.bbox.x+DELTA, ball.bbox.y+DELTA, ball.bbox.width-2*DELTA, ball.bbox.height-2*DELTA);
        ret.push_back(Ball{red_bbox, ball.type});
    }

    return ret;
}

void TrackBalls::adjustBalls(vector<int> found_indexes, vector<int> lost_indexes, vector<Rect> found_bboxes, Mat frame, vector<int>& renderer_remove_idxs) {
    
    // If the number of detected balls differs from the one of tracked balls, lost balls are erased along with their trackers
    if(balls.size() > found_bboxes.size()) {
        removeBalls(lost_indexes);
        renderer_remove_idxs = lost_indexes;
        return;
    }

    // Correctly updated balls are associated with their closest detected one, the latter is the removed from the list of found ones
    for(int i : found_indexes) {
        int relative_bbox_index = getClosestBBoxIndex(balls[i].bbox, found_bboxes);
        found_bboxes.erase(found_bboxes.begin() + relative_bbox_index);
    }

    // Wrongly updated balls are associated with their closest detected one among the list of remaing detected balls
    for(int i : lost_indexes) {
        int relative_bbox_index = getClosestBBoxIndex(balls[i].bbox, found_bboxes);
        balls[i].bbox = found_bboxes[relative_bbox_index];
    }

    // Wrongly updated balls and trackers are reinitialized with the new boxes
    for(int i : lost_indexes) multi_tracker[i]->init(frame, balls[i].bbox);

    return;
}

int TrackBalls::getClosestBBoxIndex(Rect tracked, vector<Rect> found) {
    if(found.empty()) return -1;
    // Index of the clostest detected box from the tracked one is found and returned
    float min_dist = sqEuclideanDist(tracked, found[0]);
    int index = 0;
    for(int i=1; i<found.size(); i++) {
        float dist = sqEuclideanDist(tracked, found[i]);
        index = (dist < min_dist ? i : index);
        min_dist = (dist < min_dist ? dist : min_dist);
    }

    return index;
}

float TrackBalls::sqEuclideanDist(Rect r1, Rect r2) {
    float dist = (r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y);
    return dist; 
}
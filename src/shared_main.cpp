#include <cmath>
#include <cstdlib>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <set>
#include <string>
#include <vector>
#include <algorithm>

#include "recognition/ballIdentifier.h"
#include "recognition/side_recognition.hpp"
#include "rendering/render_table.hpp"
#include "segment/segBalls.h"
#include "segment/segTable.h"
#include "tracking/TrackBalls.hpp"
#include "utils/VideoReader.hpp"
#include "utils/drawBBoxes.hpp"
#include "recognition/transformPoints.hpp"

using namespace cv;
using namespace std;

const string WINDOW_NAME = "window_main";

int main(int argc, char** argv) {

    if(argc < 2){
        cout << "Not enough arguments provided";
        exit(1);
    }

    string clip_name = argv[1];
    // cout<<clip_name<<endl;

    const string video_path = "../data/" + clip_name + "/" + clip_name + ".mp4";
    const string img_path = "../data/" + clip_name + "/frames/frame_first.png";

    Mat img_first = imread(img_path);

    VideoReader vid(video_path);
    Mat img_last = vid.lastFrame();
    Mat mask;
    vector<Point2f> points = find_table(img_last, mask);

    Mat transf = getTransformation(img_last, points);

    // Mat show;
    // warpPerspective(img_last, show, transf, img_last.size());
    // imshow(WINDOW_NAME,show);
    // waitKey(0);

    vector<Rect> bboxes = getBBoxes(img_first, mask, transf);
    drawBBoxes(img_first, bboxes);

    vector<Ball> balls = classifyBalls(img_last, bboxes);

    img_first = vid.nextFrame();
    TrackBalls tracker(img_first, balls);
    int width = img_first.cols;
    int height = img_first.rows;
    vid = VideoReader(video_path);
    TableRenderer rend(vid, tracker, balls, transf, width, height);
    while(1) {
        Mat fr = rend.nextFrame();
        if(fr.rows == 0) break;
        imshow(WINDOW_NAME, fr);
        waitKey(0);
    }

    string filename_balls = "/balls.txt";
    writeBallsFile(argv[2] + filename_balls, rend.getBalls());

    string filename_mask = "/mask.png";
    imwrite(argv[2] + filename_mask[2], mask);

    Mat layer = Mat::zeros(img_last.size(), CV_8UC3);
    vector<vector<Point>> poly_table;
    vector<Point> tmp;
    for(int i = 0; i < 4; i ++) {
        tmp.push_back(points[i]);
    }
    Mat out;
    poly_table.push_back(tmp);
    fillPoly(layer, poly_table, Scalar(0, 0, 255));
    addWeighted(img_last, 0.5, layer, 0.5, 0, out);
    imshow(WINDOW_NAME, out);
    waitKey(0);

    return 0;
}
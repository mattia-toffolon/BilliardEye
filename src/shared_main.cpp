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
#include "utils/bboxesUtils.hpp"
#include "recognition/transformPoints.hpp"

using namespace cv;
using namespace std;

const string WINDOW_NAME = "window_main";

/*
 * argument format:
 * 1)path to video mp4 file
 * 2)path to first frame png file
 * 3)path to directory to write the program outputs
 */
int main(int argc, char** argv) {

    if(argc < 4){
        cout << "Not enough arguments provided";
        exit(1);
    }

    string clip_name = argv[1];

    const string video_path = argv[1];
    const string img_path = argv[2];

    //table detection
    VideoReader vid(video_path);
    Mat img_last = vid.lastFrame();
    Mat mask;
    //vertices of the table
    vector<Point2f> points = find_table(img_last, mask);
    std::string output = argv[3];
    //since the prediction is done on the last frame we write only
    //one predicted table segmentation
    imwrite( output + "/predicted_mask.png", mask);

    //perspective transform for rendering
    Mat transf = getTransformation(img_last, points);

    //ball detection
    Mat img_first = imread(img_path);
    vector<Rect> bboxes = getBBoxes(img_first, mask, transf);

    //ball classification
    vector<Ball> balls = classifyBalls(img_first, bboxes);
    //writing the prediction of the first frame
    writeBallsFile(output + "/predicted_balls_first.txt", balls);
    imwrite(output + "/output_first.png", nice_render(img_first, points, balls));

    //creating the objects to render the video
    img_first = vid.nextFrame();
    TrackBalls tracker(img_first, balls);
    int width = 600;
    int height = 300;
    transf = getTransformation(img_last, points, width, height);
    vid = VideoReader(video_path);
    TableRenderer rend(vid, tracker, balls, transf, width, height);
    VideoWriter outvideo(output + "/render.mp4",VideoWriter::fourcc('m','p','4','v'),20, img_last.size());
    vid = VideoReader(video_path);
    //region of the minimap
    Rect spot(10, img_last.rows*2/3 - 10, img_last.rows*2/3,  img_last.rows/3);
    Mat fr;
    int i = 0;
    while(1) {
        i++;
        std::cout << i << std::endl;
        Mat curfrend = rend.nextFrame();
        if(curfrend.rows == 0) break;
        fr = curfrend;
        resize(curfrend, curfrend, spot.size());
        Mat curfr = vid.nextFrame();
        curfrend.copyTo(curfr(spot));
        outvideo.write(curfr);
    }
    outvideo.release();

    //writing the last frame of the render to have the trajectory
    const string traj = "/predicted_trajectory.png";
    imwrite(output + traj, fr);
    auto ballin = rend.getBalls();
    writeBallsFile(output + "/predicted_balls_last.txt", ballin);
    imwrite(output + "/output_last.png", nice_render(img_last, points, ballin));

    return 0;
}

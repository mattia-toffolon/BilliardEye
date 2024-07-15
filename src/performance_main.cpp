#include <iostream>
#include <string>
#include <numeric>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utils/balls.hpp>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "utils/drawBBoxes.hpp"
#include "utils/perfTesting.h"

using namespace cv;
using namespace std;

void solidTypes(const vector<Ball> &all, vector<Rect> &eight, vector<Rect> &cue, vector<Rect> &solid, vector<Rect> &striped){
    for(auto b : all){
        switch (b.type) {
            case BallType::CUE : cue.push_back(b.bbox); 
            case BallType::EIGHT : eight.push_back(b.bbox); 
            case BallType::SOLID : solid.push_back(b.bbox); 
            case BallType::STRIPED : striped.push_back(b.bbox); 
        }
    }
}
Mat drawCircles(vector<Rect> balls, Size s){
    Mat ret = Mat::zeros(s, CV_8UC1);
    for(auto r : balls){
        //drawing;
        //use function to transform rect into circle
    }
    return ret;
}
/*
 * letters in variable names:
 * f->First
 * l->Last
 * t->ground Truth
 * p->Predicted
 * ground refers to ground Truth
 * first and last refer to the respective frames
 */
//letters in variable names:
int main(int argc, char** argv) {
    if(argc < 3){
        std::cout << "not enough an arguments provided";
        exit(1);
    }
    Rect r1(0,0, 10, 10);
    Rect r2(6,6, 6, 6);
    std::cout << (r1 & r2) << std::endl;
    int samples = std::stoi(argv[2]);
    std::string directory = "/sample";
    std::string ground_mask_first = "/ground_mask_first";
    std::string ground_mask_last = "/ground_mask_last";
    std::string predicted_mask = "/predicted_mask";
    std::string predicted_ball_last = "/predicted_balls_last";
    std::string ground_balls_last = "/ground_balls_last";
    std::string predicted_ball_first = "/predicted_balls_first";
    std::string ground_balls_first = "/ground_balls_first";
    std::string txt = ".txt";
    std::string png = ".png";
    
    std::vector<Mat> ground_truth_masks_first, ground_truth_masks_last, predicted_masks;
    std::vector<std::vector<Ball>> ground_truth_balls_first, predicted_balls_first,ground_truth_balls_last, predicted_balls_last;
    for(int i = 0; i < samples; i++){
        ground_truth_masks_first.push_back(imread(argv[1] + directory + std::to_string(i) + ground_mask_first + png, IMREAD_GRAYSCALE));
        ground_truth_masks_last.push_back(imread(argv[1] + directory + std::to_string(i) + ground_mask_last + png, IMREAD_GRAYSCALE));
        predicted_masks.push_back(imread(argv[1] + directory + std::to_string(i) + predicted_mask + png, IMREAD_GRAYSCALE));
        ground_truth_balls_first.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + ground_balls_first + txt));
        predicted_balls_first.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + predicted_ball_first + txt));
        ground_truth_balls_last.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + ground_balls_last + txt));
        predicted_balls_last.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + predicted_ball_last + txt));
    }
    //insert performance computations here
    double threshold = 0.5;
    std::vector<float> precisionsf, precisionsl, precisionMaskFirst,precisionMaskLast;
    std::vector<float> ioufcue,ioufeight,ioufsolid,ioufstripe
                        ,ioulcue,iouleight,ioulsolid,ioulstripe;
    std::vector<float> ioufcueseg,ioufeightseg,ioufsolidseg,ioufstripeseg
                        ,ioulcueseg,iouleightseg,ioulsolidseg,ioulstripeseg;
    for(int i = 0; i < samples; i ++){
        std::vector<Ball> curft, curfp, curlt, curlp;
        std::vector<Rect> curftr, curfpr, curltr, curlpr;
        std::vector<Rect>
            cueft, cuefp, cuelt, cuelp,
            eightft, eightfp, eightlt, eightlp,
            stripefp, stripeft,stripelt, stripelp,
            solidft, solidfp, solidlt, solidlp;
        curft = ground_truth_balls_first[i];
        curfp = predicted_balls_first[i];
        curlt = ground_truth_balls_last[i];
        curlp = predicted_balls_last[i];
        solidTypes(curft, eightft, cueft, solidft, stripeft);
        solidTypes(curlt, eightlt, cuelt, solidlt, stripelt);
        solidTypes(curfp, eightfp, cuefp, solidfp, stripefp);
        solidTypes(curlp, eightlp, cuelp, solidlp, stripelp);

        for(int i = 0; i < curft.size(); i ++){
            curftr.push_back(curft[i].bbox);
            curfpr.push_back(curfp[i].bbox);
            curltr.push_back(curlt[i].bbox);
            curlpr.push_back(curlp[i].bbox);
        }

        drawBBoxes(imread(argv[1] + directory + std::to_string(i) + "/frame_first.png"), curftr);
        drawBBoxes(imread(argv[1] + directory + std::to_string(i) + "/frame_first.png"), curfpr);
        std::map<float, float> resf = precisionRecallCurve(curfpr, curftr, threshold);
        std::map<float, float> resl = precisionRecallCurve(curlpr, curltr, threshold);
        double curprecf =  averagePrecision(resf);
        double curprecl =  averagePrecision(resl);
        precisionsf.push_back(curprecf);
        precisionsl.push_back(curprecl);
        precisionMaskFirst.push_back(intersectionOverUnion(ground_truth_masks_first[i], predicted_masks[i]));
        precisionMaskLast.push_back(intersectionOverUnion(ground_truth_masks_last[i], predicted_masks[i]));
        //cue balls
        vector<float> cuefiou, cueliou;
        cuefiou = manyToManyIoU(cueft, cuefp);
        cueliou = manyToManyIoU(cuelt, cuelp);
        Mat cueftseg, cueltseg, cuefpseg, cuelpseg;
        cueftseg = (ground_mask_first[i] == static_cast<char>(BallType::CUE));
        cueltseg = (ground_mask_last[i] == static_cast<char>(BallType::CUE));
        cuefpseg = drawCircles(cuefp, (predicted_masks[i]).size());
        cuelpseg = drawCircles(cuelp, (predicted_masks[i]).size());
        ioufcueseg.push_back(intersectionOverUnion(cueftseg, cuefpseg));
        ioulcueseg.push_back(intersectionOverUnion(cueltseg, cuelpseg));
        ioufcue.push_back(std::accumulate(cuefiou.begin(),cuefiou.end(), 0));
        ioulcue.push_back(std::accumulate(cueliou.begin(),cueliou.end(), 0));
        //eight balls
        vector<float> eightfiou, eightliou;
        eightfiou = manyToManyIoU(eightft, eightfp);
        eightliou = manyToManyIoU(eightlt, eightlp);
        Mat eightftseg, eightltseg, eightfpseg, eightlpseg;
        eightftseg = (ground_mask_first[i] == static_cast<char>(BallType::EIGHT));
        eightltseg = (ground_mask_last[i] == static_cast<char>(BallType::EIGHT));
        eightfpseg = drawCircles(eightfp, (predicted_masks[i]).size());
        eightlpseg = drawCircles(eightlp, (predicted_masks[i]).size());
        ioufeightseg.push_back(intersectionOverUnion(eightftseg, eightfpseg));
        iouleightseg.push_back(intersectionOverUnion(eightltseg, eightlpseg));
        ioufeight.push_back(std::accumulate(eightfiou.begin(),eightfiou.end(), 0));
        iouleight.push_back(std::accumulate(eightliou.begin(),eightliou.end(), 0));
        //solid balls
        vector<float> solidfiou, solidliou;
        solidfiou = manyToManyIoU(solidft, solidfp);
        solidliou = manyToManyIoU(solidlt, solidlp);
        Mat solidftseg, solidltseg, solidfpseg, solidlpseg;
        solidftseg = (ground_mask_first[i] == static_cast<char>(BallType::SOLID));
        solidltseg = (ground_mask_last[i] == static_cast<char>(BallType::SOLID));
        solidfpseg = drawCircles(solidfp, (predicted_masks[i]).size());
        solidlpseg = drawCircles(solidlp, (predicted_masks[i]).size());
        ioufsolidseg.push_back(intersectionOverUnion(solidftseg, solidfpseg));
        ioulsolidseg.push_back(intersectionOverUnion(solidltseg, solidlpseg));
        ioufsolid.push_back(std::accumulate(solidfiou.begin(),solidfiou.end(), 0));
        ioulsolid.push_back(std::accumulate(solidliou.begin(),solidliou.end(), 0));
        //striped
        vector<float> stripefiou, stripeliou;
        stripefiou = manyToManyIoU(stripeft, stripefp);
        stripeliou = manyToManyIoU(stripelt, stripelp);
        Mat stripedftseg, stripedltseg, stripedfpseg, stripedlpseg;
        stripedftseg = (ground_mask_first[i] == static_cast<char>(BallType::STRIPED));
        stripedltseg = (ground_mask_last[i] == static_cast<char>(BallType::STRIPED));
        stripedfpseg = drawCircles(stripefp, (predicted_masks[i]).size());
        stripedlpseg = drawCircles(stripelp, (predicted_masks[i]).size());
        ioufstripeseg.push_back(intersectionOverUnion(stripedftseg, stripedfpseg));
        ioulstripeseg.push_back(intersectionOverUnion(stripedltseg, stripedlpseg));
        ioufstripe.push_back(std::accumulate(stripefiou.begin(),stripefiou.end(), 0));
        ioulstripe.push_back(std::accumulate(stripeliou.begin(),stripeliou.end(), 0));
    }
    for(int i = 0; i < samples; i ++){
        std::cout << "precision first " << precisionsf[i] << std::endl;
        std::cout << "precision last " << precisionsl[i] << std::endl;
        std::cout << "precision mask first" << precisionMaskFirst[i] << std::endl;
        std::cout << "precision mask last" << precisionMaskLast[i] << std::endl;
    }
}

#include <iostream>
#include <string>
#include <numeric>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utils/balls.hpp>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "utils/bboxesUtils.hpp"
#include "utils/perfTesting.h"

using namespace cv;
using namespace std;

void fillTypes(const vector<Ball> &all, vector<Rect> &eight, vector<Rect> &cue, vector<Rect> &solid, vector<Rect> &striped){
    std::cout << "AAAAAAAAA\n";
    for(auto b : all){
        switch (b.type) {
            case BallType::CUE : cue.push_back(b.bbox); break;
            case BallType::EIGHT : std::cout << "yippeee\n";eight.push_back(b.bbox); break;
            case BallType::SOLID : solid.push_back(b.bbox); break;
            case BallType::STRIPED : striped.push_back(b.bbox); break;
        }
    }
}
Mat drawCircles(vector<Rect> balls, Size s, Scalar color = Scalar(255)){
    Mat ret = Mat::zeros(s, CV_8UC1);
    for(auto r : balls){
        Vec3f circ = toCircle(r);
        circle(ret, Point(circ[0],circ[1]), circ[2], color, FILLED);
    }
    return ret;
}
Mat drawCircles(vector<Ball> balls, Size s, Scalar color, Mat initial){
    Mat ret = initial.clone();
    for(auto r : balls){
        Vec3f circ = toCircle(r.bbox);
        circle(ret, Point(circ[0],circ[1]), circ[2], color, FILLED);
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
        for(auto b : curfp){
            std::cout << b.bbox << b.type << std::endl;
        }
        curlt = ground_truth_balls_last[i];
        curlp = predicted_balls_last[i];
        fillTypes(curft, eightft, cueft, solidft, stripeft);
        fillTypes(curlt, eightlt, cuelt, solidlt, stripelt);
        fillTypes(curfp, eightfp, cuefp, solidfp, stripefp);
        fillTypes(curlp, eightlp, cuelp, solidlp, stripelp);

        for(int i = 0; i < curft.size(); i ++){
            curftr.push_back(curft[i].bbox);
        }
        for(int i = 0; i < curfp.size(); i ++){
            curfpr.push_back(curfp[i].bbox);
        }
        for(int i = 0; i < curlt.size(); i ++){
            curltr.push_back(curlt[i].bbox);
        }
        for(int i = 0; i < curlp.size(); i ++){
            curlpr.push_back(curlp[i].bbox);
        }

        //drawBBoxes(imread(argv[1] + directory + std::to_string(i) + "/frame_last.png"), curltr);
        //drawBBoxes(imread(argv[1] + directory + std::to_string(i) + "/frame_last.png"), curlpr);
        std::map<float, float> resf = precisionRecallCurve(curfpr, curftr, threshold);
        std::map<float, float> resl = precisionRecallCurve(curlpr, curltr, threshold);
        double curprecf =  averagePrecision(resf);
        double curprecl =  averagePrecision(resl);
        precisionsf.push_back(curprecf);
        precisionsl.push_back(curprecl);
        Mat predicted_mask_first, predicted_mask_last;
        predicted_mask_first = drawCircles(curfp, predicted_masks[i].size(), Scalar(0),predicted_masks[i]);
        predicted_mask_last = drawCircles(curlp, predicted_masks[i].size(), Scalar(0),predicted_masks[i]);
        precisionMaskFirst.push_back(intersectionOverUnion(ground_truth_masks_first[i] ==5, predicted_mask_first));
        precisionMaskLast.push_back(intersectionOverUnion(ground_truth_masks_last[i] == 5, predicted_mask_last));
        //cue balls
        vector<float> cuefiou, cueliou;
        cuefiou = manyToManyIoU(cueft, cuefp);
        cueliou = manyToManyIoU(cuelt, cuelp);
        Mat cueftseg, cueltseg, cuefpseg, cuelpseg;
        cueftseg = (ground_truth_masks_first[i] == static_cast<char>(BallType::CUE));
        cueltseg = (ground_truth_masks_last[i] == static_cast<char>(BallType::CUE));
        cuefpseg = drawCircles(cuefp, (predicted_masks[i]).size());
        cuelpseg = drawCircles(cuelp, (predicted_masks[i]).size());
        ioufcueseg.push_back(intersectionOverUnion(cueftseg, cuefpseg));
        ioulcueseg.push_back(intersectionOverUnion(cueltseg, cuelpseg));
        ioufcue.push_back(std::accumulate(cuefiou.begin(),cuefiou.end(), 0.0)/static_cast<float>(cuefiou.size()));
        ioulcue.push_back(std::accumulate(cueliou.begin(),cueliou.end(), 0.0)/static_cast<float>(cueliou.size()));
        //eight balls
        vector<float> eightfiou, eightliou;
        eightfiou = manyToManyIoU(eightft, eightfp);
        eightliou = manyToManyIoU(eightlt, eightlp);
        Mat eightftseg, eightltseg, eightfpseg, eightlpseg;
        eightftseg = (ground_truth_masks_first[i] == static_cast<char>(BallType::EIGHT));
        //imshow("eight true", eightftseg);
        eightltseg = (ground_truth_masks_last[i] == static_cast<char>(BallType::EIGHT));
        eightfpseg = drawCircles(eightfp, (predicted_masks[i]).size());
        //imshow("eight predicted", eightfpseg);
        //imshow("eight inter", (eightfpseg & eightftseg));
        //waitKey(0);
        eightlpseg = drawCircles(eightlp, (predicted_masks[i]).size());
        ioufeightseg.push_back(intersectionOverUnion(eightftseg, eightfpseg));
        iouleightseg.push_back(intersectionOverUnion(eightltseg, eightlpseg));
        ioufeight.push_back(std::accumulate(eightfiou.begin(),eightfiou.end(), 0.0)/static_cast<float>(eightfiou.size()));
        iouleight.push_back(std::accumulate(eightliou.begin(),eightliou.end(), 0.0)/static_cast<float>(eightliou.size()));
        //solid balls
        vector<float> solidfiou, solidliou;
        solidfiou = manyToManyIoU(solidft, solidfp);
        solidliou = manyToManyIoU(solidlt, solidlp);
        Mat solidftseg, solidltseg, solidfpseg, solidlpseg;
        solidftseg = (ground_truth_masks_first[i] == static_cast<char>(BallType::SOLID));
        solidltseg = (ground_truth_masks_last[i] == static_cast<char>(BallType::SOLID));
        solidfpseg = drawCircles(solidfp, (predicted_masks[i]).size());
        solidlpseg = drawCircles(solidlp, (predicted_masks[i]).size());
        ioufsolidseg.push_back(intersectionOverUnion(solidftseg, solidfpseg));
        ioulsolidseg.push_back(intersectionOverUnion(solidltseg, solidlpseg));
        ioufsolid.push_back(std::accumulate(solidfiou.begin(),solidfiou.end(), 0.0)/static_cast<float>(solidfiou.size()));
        ioulsolid.push_back(std::accumulate(solidliou.begin(),solidliou.end(), 0.0)/static_cast<float>(solidliou.size()));
        //striped
        vector<float> stripefiou, stripeliou;
        stripefiou = manyToManyIoU(stripeft, stripefp);
        stripeliou = manyToManyIoU(stripelt, stripelp);
        Mat stripedftseg, stripedltseg, stripedfpseg, stripedlpseg;
        stripedftseg = (ground_truth_masks_first[i] == static_cast<char>(BallType::STRIPED));
        stripedltseg = (ground_truth_masks_last[i] == static_cast<char>(BallType::STRIPED));
        stripedfpseg = drawCircles(stripefp, (predicted_masks[i]).size());
        stripedlpseg = drawCircles(stripelp, (predicted_masks[i]).size());
        ioufstripeseg.push_back(intersectionOverUnion(stripedftseg, stripedfpseg));
        ioulstripeseg.push_back(intersectionOverUnion(stripedltseg, stripedlpseg));
        ioufstripe.push_back(std::accumulate(stripefiou.begin(),stripefiou.end(), 0.0)/static_cast<float>(stripefiou.size()));
        ioulstripe.push_back(std::accumulate(stripeliou.begin(),stripeliou.end(), 0.0)/static_cast<float>(stripeliou.size()));
    }
    for(int i = 0; i < samples; i ++){
        std::cout << "sample " << i << std::endl;
        std::cout << "mean average precision first " << precisionsf[i] << std::endl;
        std::cout << "mean average precision last " << precisionsl[i] << std::endl;
        std::cout << "mean iou bbox first cue " << ioufcue[i] << std::endl;
        std::cout << "mean iou bbox last cue " << ioulcue[i] << std::endl;
        std::cout << "mean iou bbox first eight " << ioufeight[i] << std::endl;
        std::cout << "mean iou bbox last eight " << iouleight[i] << std::endl;
        std::cout << "mean iou bbox first solid " << ioufsolid[i] << std::endl;
        std::cout << "mean iou bbox last solid " << ioulsolid[i] << std::endl;
        std::cout << "mean iou bbox first stripe " << ioufstripe[i] << std::endl;
        std::cout << "mean iou bbox last stripe " << ioulstripe[i] << std::endl;
        std::cout << "mean iou segmentation first cue " << ioufcueseg[i] << std::endl;
        std::cout << "mean iou segmentation last cue " << ioulcueseg[i] << std::endl;
        std::cout << "mean iou segmentation first eight " << ioufeightseg[i] << std::endl;
        std::cout << "mean iou segmentation last eight " << iouleightseg[i] << std::endl;
        std::cout << "mean iou segmentation first solid " << ioufsolidseg[i] << std::endl;
        std::cout << "mean iou segmentation last solid " << ioulsolidseg[i] << std::endl;
        std::cout << "mean iou segmentation first stripe " << ioufstripeseg[i] << std::endl;
        std::cout << "mean iou segmentation last stripe " << ioulstripeseg[i] << std::endl;
        std::cout << "precision mask first" << precisionMaskFirst[i] << std::endl;
        std::cout << "precision mask last" << precisionMaskLast[i] << std::endl;
        std::cout << std::endl;
    }
}

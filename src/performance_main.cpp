#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utils/balls.hpp>
#include <string>

using namespace cv;

int main(int argc, char** argv) {
    if(argc < 3){
        std::cout << "not enough an arguments provided";
        exit(1);
    }
    int samples = std::stoi(argv[2]);
    std::string directory = "/sample";
    std::string ground_mask = "/ground_mask";
    std::string predicted_mask = "/predicted_mask";
    std::string predicted_ball = "/predicted_balls";
    std::string ground_balls = "/ground_balls";
    std::string txt = ".txt";
    std::string png = ".png";
    
    std::vector<Mat> ground_truth_masks, predicted_masks;
    std::vector<std::vector<Ball>> ground_truth_balls, predicted_balls;
    for(int i = 0; i < samples; i++){
        ground_truth_masks.push_back(imread(argv[1] + directory + std::to_string(i) + ground_mask + png));
        predicted_masks.push_back(imread(argv[1] + directory + std::to_string(i) + predicted_mask + png));
        ground_truth_balls.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + ground_balls + txt));
        predicted_balls.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + predicted_ball + txt));
    }
    //insert performance computations here
}

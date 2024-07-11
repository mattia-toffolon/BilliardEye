#ifndef RENDER_TABLE
#define RENDER_TABLE
#include "tracking/TrackBalls.hpp"
#include "utils/VideoReader.hpp"
#include <vector>
class TableRenderer{
    VideoReader vid;
    TrackBalls tracker;
    cv::Mat curimg;
    std::vector<Ball> bbs;
    std::vector<cv::Point> holes;
    float hole_radius = 20;
    cv::Mat transform;
    public:
    /**
     * @param v 
     * @param t
     * @param balls  
     * @param transform perspective transform of the table 
     * @param width width of the dst image (the same as the perspective transform)  
     * @param height height of the dst image (the same as the perspective transform)  
     * 
     */
    TableRenderer(VideoReader v, TrackBalls t, std::vector<Ball> balls, cv::Mat transform, int width, int cols);
    /*
     * return the next frame of the rendered table
     */
    cv::Mat nextFrame();
    /*
     * return the vector of the current balls objets
     */
    std::vector<Ball> getBalls();
    private:
    bool is_holed(cv::Point ball);
};
cv::Mat nice_render(cv::Mat);
#endif

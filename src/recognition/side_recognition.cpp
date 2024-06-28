#include "recognition/side_recognition.hpp"
using namespace cv;
std::vector<Mat> getRotatedborders(const std::vector<Point2f> points, const Mat img, int thick){
    std::vector<Mat> ret;
    for(int i = 0; i < 4; i++){
        Point p1, p2;
        if(i<3){
            p1 = points[i];
            p2 = points[i+1];
        }else{
            p1 = points[i];
            p2 = points[0];
        }
        std::vector<Point> pts;
        pts.push_back(p1);
        pts.push_back(p2);
        RotatedRect rect = minAreaRect(pts);
        if(rect.size.width)
            rect.size.height = thick;
        else
            rect.size.width = thick;

        if (rect.size.width)
            rect.size.height = thick;
        else
            rect.size.width = thick;
        std::vector<Point2f> boundingpoints;
        rect.points(boundingpoints);
        Mat M, rotated, cropped;
        float angle = rect.angle;
        Size rect_size = rect.size;
        if (rect.angle < -45.) {
            angle += 90.0;
            swap(rect_size.width, rect_size.height);
        }
        M = getRotationMatrix2D(rect.center, angle, 1.0);
        warpAffine(img, rotated, M, img.size(), INTER_CUBIC);
        getRectSubPix(rotated, rect_size, rect.center, cropped);
        ret.push_back(cropped);

    }
    return ret;

}
bool isShortFirst(std::vector<cv::Mat> sides){
    double acc[]{0,0};

    for(int i = 0; i < 4; i++){
        Mat cropped = sides[i]; 
        if(cropped.rows > cropped.cols){
            cropped = cropped.t();
        }
        int corner = 30;
        Mat sec1, sec2, sec3;
        Sobel(cropped, cropped, CV_8U, 1, 0);
        Scalar tmpacc = sum(cropped(Rect(cropped.cols/4, 0, cropped.cols/2, cropped.rows)))/(cropped.rows*cropped.cols/2);
        acc[i%2]+=tmpacc[0];
    }
    return acc[0]<acc[1];
}

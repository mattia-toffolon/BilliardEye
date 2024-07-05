#include "utils/perfTesting.h"

using namespace cv;
using std::vector;
using std::map;

float intersectionOverUnion(Rect region1, Rect region2)
{
    float areaInters = (region1 & region2).area();
    float areaUnion = region1.area() + region2.area() - areaInters;
    return areaInters / areaUnion;
}

float intersectionOverUnion(Mat mask1, Mat mask2)
{
    assert(mask1.size() == mask2.size());
    Mat intersection;
    bitwise_and(mask1,mask2,intersection);
    float areaInters = countNonZero(intersection);
    float areaUnion = countNonZero(mask1) + countNonZero(mask2) - areaInters;
    return areaInters/areaUnion;
}

float oneToManyIoU(Rect region, vector<Rect> candidates)
{
    float best = 0;
    for (Rect r : candidates)
    {
        best = std::max(best,intersectionOverUnion(region,r));
    }
    return best;
}

vector<float> manyToManyIoU(vector<Rect> regions1, vector<Rect> regions2)
{
    vector<float> ans;
    for (Rect r1 : regions1)
    {
        ans.push_back(oneToManyIoU(r1,regions2));
    }
    return ans;
}

map<float,float> precisionRecallCurve(
    vector<Rect> regions1, 
    vector<Rect> regions2,
    float threshold=0.5
)
{
    map<float,float> ans;
    // TODO
}
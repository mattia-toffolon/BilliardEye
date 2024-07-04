#include "utils/perfTesting.h"

using namespace cv;

float intersectionOverUnion(Rect region1, Rect region2)
{
    float areaInters = (region1 & region2).area();
    float areaUnion = region1.area() + region2.area() - areaInters;
    return areaInters / areaUnion;
}
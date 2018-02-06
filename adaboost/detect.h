#ifndef DETECT_H
#define DETECT_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "classifier.h"
#include "train.h"

const double scale_coef(1.4);
const int step(2);

struct Area{
    CvPoint leftTop;
    CvPoint rightBottom;
    int count;
    CvPoint leye;
    CvPoint reye;
};

void normalizePixel(std::vector<std::vector<double>> &pixTable, int width, int height);
void getPixData(IplImage *img, std::vector<std::vector<double>> &pixTable);
void ccalcIntegralImage(std::vector<std::vector<double>> &pixTable, std::vector<std::vector<double>> &ii);
void getIntegralImage(std::vector<std::vector<double>> &ii, double cii[regWidth][regHeight], int w, int h);
int detectFace(IplImage* img, CascadeClassifier &classifier);
bool faceExist(std::vector<Area> &faces, Area &face);
int detectArea(std::vector<Area> &faces, IplImage* img, CvRect &rect, CascadeClassifier &cc);
void areaCombine(std::vector<Area> &faces);
void colorDetect(std::vector<Area> &faces, IplImage *img);
bool findEye(Area area, IplImage* img);
void equalizeHist(IplImage* img);

#endif // DETECT_H

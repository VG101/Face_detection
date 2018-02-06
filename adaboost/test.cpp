#include "train.h"
#include "classifier.h"
#include "detect.h"
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

using std::vector;
using std::string;
using std::stringstream;
using std::cout;

int main()
{
    CascadeClassifier cc;
    FILE *fp;
    int ssize;
    fp = fopen("classifiers/cascade_classifier_1", "rb");
    fread(&ssize, sizeof(int), 1, fp);
    StrongClassifier sc1;
    for (int i = 0; i < ssize; ++i){
        WeakClassifier wc;
        fread(&wc, sizeof(WeakClassifier), 1, fp);
        sc1.wClassifiers.push_back(wc);
    }
    setSCThreshold(sc1, 6.0);
    fclose(fp);
    cc.sClassifiers.push_back(sc1);

    fp = fopen("classifiers/cascade_classifier_2", "rb");
    fread(&ssize, sizeof(int), 1, fp);
    StrongClassifier sc2;
    for (int i = 0; i < ssize; ++i){
        WeakClassifier wc;
        fread(&wc, sizeof(WeakClassifier), 1, fp);
        sc2.wClassifiers.push_back(wc);
    }
    setSCThreshold(sc2, 9.0);
    fclose(fp);
    cc.sClassifiers.push_back(sc2);

    fp = fopen("classifiers/cascade_classifier_3", "rb");
    fread(&ssize, sizeof(int), 1, fp);
    StrongClassifier sc3;
    for (int i = 0; i < ssize; ++i){
        WeakClassifier wc;
        fread(&wc, sizeof(WeakClassifier), 1, fp);
        sc3.wClassifiers.push_back(wc);
    }
    setSCThreshold(sc3, 6.4);
    fclose(fp);
    cc.sClassifiers.push_back(sc3);

    fp = fopen("classifiers/cascade_classifier_4", "rb");
    fread(&ssize, sizeof(int), 1, fp);
    StrongClassifier sc4;
    for (int i = 0; i < ssize; ++i){
        WeakClassifier wc;
        fread(&wc, sizeof(WeakClassifier), 1, fp);
        sc4.wClassifiers.push_back(wc);
    }
    setSCThreshold(sc4, 3.4);
    fclose(fp);
    cc.sClassifiers.push_back(sc4);

    fp = fopen("classifiers/cascade_classifier_5", "rb");
    fread(&ssize, sizeof(int), 1, fp);
    StrongClassifier sc5;
    for (int i = 0; i < ssize; ++i){
        WeakClassifier wc;
        fread(&wc, sizeof(WeakClassifier), 1, fp);
        sc5.wClassifiers.push_back(wc);
    }
    setSCThreshold(sc5, 3.9);
    fclose(fp);
    cc.sClassifiers.push_back(sc5);

    fp = fopen("classifiers/cascade_classifier_6", "rb");
    fread(&ssize, sizeof(int), 1, fp);
    StrongClassifier sc6;
    for (int i = 0; i < ssize; ++i){
        WeakClassifier wc;
        fread(&wc, sizeof(WeakClassifier), 1, fp);
        sc6.wClassifiers.push_back(wc);
    }
    setSCThreshold(sc6, 2.8);
    fclose(fp);
    cc.sClassifiers.push_back(sc6);

    fp = fopen("classifiers/cascade_classifier_7", "rb");
    fread(&ssize, sizeof(int), 1, fp);
    StrongClassifier sc7;
    for (int i = 0; i < ssize; ++i){
        WeakClassifier wc;
        fread(&wc, sizeof(WeakClassifier), 1, fp);
        sc7.wClassifiers.push_back(wc);
    }
    setSCThreshold(sc7, 2.8);
    fclose(fp);
    cc.sClassifiers.push_back(sc7);

    fp = fopen("classifiers/cascade_classifier_8", "rb");
    fread(&ssize, sizeof(int), 1, fp);
    StrongClassifier sc8;
    for (int i = 0; i < ssize; ++i){
        WeakClassifier wc;
        fread(&wc, sizeof(WeakClassifier), 1, fp);
        sc8.wClassifiers.push_back(wc);
    }
    setSCThreshold(sc8, 2.6);
    fclose(fp);
    cc.sClassifiers.push_back(sc8);

    fp = fopen("classifiers/cascade_classifier_9", "rb");
    fread(&ssize, sizeof(int), 1, fp);
    StrongClassifier sc9;
    for (int i = 0; i < ssize; ++i){
        WeakClassifier wc;
        fread(&wc, sizeof(WeakClassifier), 1, fp);
        sc9.wClassifiers.push_back(wc);
    }
    setSCThreshold(sc9, 2.0);
    fclose(fp);
    cc.sClassifiers.push_back(sc9);

    IplImage *img = cvLoadImage("faces/image_0124.jpg");
    CvSize csize = {img->width * 0.3, img->height * 0.3};
    IplImage *cimg = cvCreateImage(csize, img->depth, img->nChannels);
    cvResize(img, cimg, CV_INTER_AREA);
    cout << detectFace(cimg, cc);

}

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
    setSCThreshold(sc1, 6.5);
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
    setSCThreshold(sc9, 1.9);
    fclose(fp);
    cc.sClassifiers.push_back(sc9);

    /*fp = fopen("classifiers/cascade_classifier_10", "rb");
    fread(&ssize, sizeof(int), 1, fp);
    StrongClassifier sc10;
    for (int i = 0; i <= ssize; ++i){
        WeakClassifier wc;
        fread(&wc, sizeof(WeakClassifier), 1, fp);
        //cout << i << ": " << wc.weight << "\n";
        sc10.wClassifiers.push_back(wc);
    }
    setSCThreshold(sc10, 3.7);
    fclose(fp);
    cc.sClassifiers.push_back(sc10);*/

    int face1 = 0;
    int feret = 1400;
    int cface = 0;
    int face = feret + face1 + cface;
    int nface = 2000;
    vector<TrainSample> test;
    string face1Dir = "test/500/cc (";
    for(int i =1; i <= face1;i++){
        stringstream ss;
        ss<<face1Dir<<i<<").jpg";
        addImg(test, ss.str(), true);
    }
    string faceDir = "train/cface/cf (";
    for (int i = 1; i <= cface; ++ i){
        stringstream ss;
        ss << faceDir << i << ").jpg";
        addImg(test, ss.str(), true);
    }
    string feretDir = "train/FERET/FERET-00";
    for (int i = 1; i <= 200; ++ i){
        if (i == 10)
            feretDir = "train/FERET/FERET-0";
        else if (i == 100)
            feretDir = "train/FERET/FERET-";
        for (int j = 1; j <= 7; ++ j){
            stringstream ss;
            ss << feretDir << i <<"/0" << j << ".tif";
            addImg(test, ss.str(), true);
        }
    }
    string nfaceDir = "test/non-face/cmu_000";
    for(int i=0; i<nface;i++){
        if (i == 10)
            nfaceDir = "test/non-face/cmu_00";
        else if (i == 100)
            nfaceDir = "test/non-face/cmu_0";
        else if (i == 1000)
            nfaceDir = "test/non-face/cmu_";
        stringstream ss;
        ss<<nfaceDir<<i<<".pgm";
        addImg(test,ss.str(),false);
    }

    /*FILE *tfp;
    int tsize;
    for (int t = 1; t <= 2; ++t){
        stringstream ss;
        ss << "ts/t" << t;
        tfp = fopen(ss.str().c_str(), "rb");
        fread(&tsize, sizeof(int), 1, tfp);
        for (int i = 0; i < tsize; ++ i){
            TrainSample ts;
            fread(&ts, sizeof(TrainSample), 1, tfp);
            test.push_back(ts);
        }
        fclose(tfp);
    }*/

    int d = 0;
    int f = 0;
    for (int i = 0; i <test.size();++i){
        if (cascadeClassify(test[i], cc) == true && test[i].isFace == false)
            f++;
        else if (cascadeClassify(test[i], cc) == false && test[i].isFace == true){
            d++;
            //cout << i + 1<<"\n";
        }
    }

    cout << "There are "<<test.size()<<" test cases\n";
    cout << "The correct rate is "<< 100*(double)(face-d)/face <<"%\n";
    cout << d << "\n";
    cout << "The wrong rate is "<< 100*(double)(f)/(test.size()-face) <<"%\n";
	
	return 0;
}
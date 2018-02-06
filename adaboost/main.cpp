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
    setSCThreshold(sc9, 2.0);
    fclose(fp);
    cc.sClassifiers.push_back(sc9);



    int cface = 0;
    int feretface = 1400;
    int face = cface + feretface;
    int b1 = 500;
    int b20 = 2500;
    int gulf = 165;
    int goldwater = 227;
    int graves = 227;
    int geyser = 227;
    int n1 = 814;
    int cars = 516;
    int b5 = 340;
    int background = 550;
    int n = 274;
    vector<TrainSample> samples;
    string feretDir = "train/FERET/FERET-00";
    for (int i = 1; i <= 200; ++ i){
        if (i == 10)
            feretDir = "train/FERET/FERET-0";
        else if (i == 100)
            feretDir = "train/FERET/FERET-";
        for (int j = 1; j <= 7; ++ j){
            stringstream ss;
            ss << feretDir << i <<"/0" << j << ".tif";
            addImg(samples, ss.str(), true);
        }
    }

    /*string b1Dir = "train/non-face/B1_0000";
    for (int i = 1; i <= b1; ++ i){
        if (i == 10)
            b1Dir = "train/non-face/B1_000";
        else if (i == 100)
            b1Dir = "train/non-face/B1_00";
        stringstream ss;
        ss << b1Dir << i <<".pgm";
        addImg(samples, ss.str(), false);
    }
    string b5Dir = "train/non-face/B5_0000";
    for (int i = 0; i <= b5; ++ i){
        if (i == 10)
            b5Dir = "train/non-face/B5_000";
        if (i == 100)
            b5Dir = "train/non-face/B5_00";
        stringstream ss;
        ss << b5Dir << i << ".pgm";
        addImg(samples, ss.str(), false);
    }
    string b20Dir = "train/non-face/B20_0";
    for (int i = 1507; i <= (b20 + 1507); ++ i){
        stringstream ss;
        ss << b20Dir << i << ".pgm";
        addImg(samples, ss.str(), false);
    }
    string gulfDir = "train/non-face/GULF_";
    for (int i = 0; i < gulf; ++ i){
        stringstream ss;
        ss << gulfDir << i << ".pgm";
        addImg(samples, ss.str(), false);
    }
    string goldwaterDir = "train/non-face/goldwater67_";
    for (int i = 0; i <= goldwater; ++ i){
        stringstream ss;
        ss << goldwaterDir << i << ".pgm";
        addImg(samples, ss.str(), false);
    }
    string gravesDir = "train/non-face/graves111_";
    for (int i = 0; i <= graves; ++ i){
        stringstream ss;
        ss << gravesDir << i << ".pgm";
        addImg(samples, ss.str(), false);
    }
    string geyserDir = "train/non-face/geyser27_";
    for (int i = 0; i <= geyser; ++ i){
        stringstream ss;
        ss << geyserDir << i << ".pgm";
        addImg(samples, ss.str(), false);
    }
    string n1Dir = "train/n1/n1 (";
    for (int i = 1; i <= n1; ++ i){
        stringstream ss;
        ss << n1Dir << i << ").bmp";
        addImg(samples, ss.str(), false);
    }
    string backgroundDir = "train/background/0";
    for (int i = 100; i <= 550; ++ i){
        stringstream ss;
        ss << backgroundDir << i << ".jpg";
        addSamples(samples, ss.str());
    }
    string n2Dir = "train/n2/n (";
    for (int i = 1; i <= 274; ++ i){
        stringstream ss;
        ss << n2Dir << i << ").jpg";
        addSamples(samples, ss.str(), cc);
    }
    string nfaceDir = "test/non-face/cmu_000";
    for(int i=0; i<3500;i++){
        if (i == 10)
            nfaceDir = "test/non-face/cmu_00";
        else if (i == 100)
            nfaceDir = "test/non-face/cmu_0";
        else if (i == 1000)
            nfaceDir = "test/non-face/cmu_";
        stringstream ss;
        ss<<nfaceDir<<i<<".pgm";
        addImg(samples,ss.str(),false);
    }*/
    /*string newDir = "train/new/n (";
    for (int i = 1; i <= 141; ++ i){
        stringstream ss;
        ss << newDir << i << ").jpg";
        addSamples(samples, ss.str(), cc);
    }
    for (int i = 1; i <= 35; ++i){
        stringstream ss;
        ss << "train/nn/n ("<< i <<").jpg";
        addSamples(samples, ss.str(), cc);
    }*/

    //cout << "total samples: " << samples.size()<<"\n";
    //cout << "positve samples: "<<face<<", negative samples: "<< samples.size()-face<<"\n";



    /*FILE *tfp;
    tfp = fopen("ts/t2", "wb");
    int tsize = samples.size();
    fwrite(&tsize, sizeof(int), 1, tfp);
    for (int i = 0; i < tsize; ++ i){
        fwrite(&samples[i], sizeof(TrainSample), 1, tfp);
    }
    fclose(tfp);*/

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
            samples.push_back(ts);
        }
        fclose(tfp);
    }*/

    cout << "total samples: " << samples.size()<<"\n";
    cout << "positve samples: "<<face<<", negative samples: "<< samples.size()-face<<"\n";

    initializeWeight(samples, face, samples.size()-face);

    vector<Feature> features;
    //initializeFeatures(features);
    FILE *ffp;
    ffp = fopen("features/feature2", "rb");
    int fsize;
    fread(&fsize, sizeof(int), 1, ffp);
    for (int i = 0; i < fsize; ++ i){
        Feature f;
        fread(&f, sizeof(Feature), 1, ffp);
        features.push_back(f);
    }
    fclose(ffp);
    cout << "feature:"<<features.size()<<"\n";

    StrongClassifier sc = trainStrongClassifier(samples, features, 50);

    FILE *fp1;
    fp1 = fopen("classifiers/cascade_classifier_10", "wb");
    int si = 50;
    fwrite(&si, sizeof(int), 1, fp1);
    for (int i = 0; i < si; i ++){
        fwrite(&sc.wClassifiers[i], sizeof(WeakClassifier), 1, fp1);
    }
    fclose(fp1);


    return 0;
}

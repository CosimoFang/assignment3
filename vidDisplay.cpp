//
//  vidDisplay.cpp
//  Created by Hongchao Fang on 10/22/23.
//


#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <sstream>  

#include <tuple>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "dirent.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core/core.hpp"

#include "filter.h"

#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;

//store the feature into db.txt file
int storeFeatures(std::string name, std::vector<float> features) {
    for (float a : features) {
        name += ", " + std::to_string(a);
    }
    std::ofstream fout;
    fout.open("db.txt", std::ios_base::app);
    if (fout.is_open()) {
        fout << name << std::endl;
        std::cout << "saved" << std::endl;
        fout.close();

    }
    return 0;
}


//read feature from db.txt file
std::vector<std::tuple<std::string, std::vector<float>>> loadFeatures() {
    std::fstream newfile;
    newfile.open("db.txt", std::ios::in);
    std::vector<std::tuple<std::string, std::vector<float>>> features;
    if (newfile.is_open()) { 
        std:: string tp;
        while (getline(newfile, tp)) { 
            std::string name;
            std::vector<float> result;
            std::stringstream X(tp);
            std::string T;
            while (std::getline(X, T, ',')) {
                if (name.empty()) {
                    name = T;
                }
                else {
                    result.push_back(std::stof(T));
                }
                
            }
            features.push_back(std::make_tuple(name, result));

        }


        newfile.close(); 
    }

    return features;
}


int main(int argc, char* argv[]) {
    cv::VideoCapture* capdev;
    std::vector<std::tuple<std::string, std::vector<float>>> db = loadFeatures();
    int k = 3;
    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);
    printf("Press L to load new feature\n");
    printf("Press Q to quit\n");

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;
    int count = 0;
    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }
        cv::Mat thresholded_frame = thresholded(frame);
        cv::Mat morphological_frame = morphological(thresholded_frame);
        cv::Mat connected_frame;
        std::vector<float> features = compute_features(morphological_frame, connected_frame);
        //std::string featuresToDisplay = findObject(db, features);
        std::string featuresToDisplay = kNearestNeighbor(k, db, features);
        if (featuresToDisplay == "UNKNOWN") {
            count += 1;
        } else {
            count = 0;
        }
        cv::putText(connected_frame, featuresToDisplay, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(54, 117, 136), 2);





        cv::imshow("Video", connected_frame);

        // see if there is a waiting keystroke
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        //reset the database to empty
        /*if (key == 'r') {
            std::ofstream ofs;
            ofs.open("db.txt", std::ofstream::out | std::ofstream::trunc);
            ofs.close();
            db = loadFeatures();
        }*/
        //load this img as new feature when l is pressed or img is unknown for 300 frames
        if (key == 'l' or count >= 300) {
            count = 0;
            std::string mystr;
            std::cout << "What's your object name? ";
            getline(std::cin, mystr);
            storeFeatures(mystr, features);
            db = loadFeatures();
        }
    }

    delete capdev;
    return(0);
}

std::string kNearestNeighbor(int k, cv::String featureVectorStr)
{

    std::map<std::string, std::vector<std::string>> vecmap;

    // get the Name : dist met from the string
    // https://www.geeksforgeeks.org/how-to-insert-data-in-the-map-of-strings/
    std::ifstream myfile;
    myfile.open("DistanceMetrics.txt");
    std::string myline;
    if (myfile.is_open())
    {
        while (myfile)
        {
            std::getline(myfile, myline);
            std::string s = myline;
            if (s.length() < 10)
            {
                break;
            }
            std::string category = s.substr(0, s.find_first_of(","));
            std::string distMetString = s.substr(s.find_last_of(", "), s.length());

            if (atof(distMetString.c_str()) == 0)
            {
                continue;
            }

            // add to the map --> name : distances, update if found
            if (vecmap.find(category) != vecmap.end())
            {
                auto cat = vecmap.find(category);
                std::vector<std::string> currentVec = cat->second;
                currentVec.push_back(distMetString);
                cat->second = currentVec;
            }
            else
            {
                std::vector<std::string> v = {distMetString};
                vecmap.insert(std::pair<std::string, std::vector<std::string>>(category, v));
            }
        }
    }

    double smallest = 100;
    std::string smallestCategory = "";
    // go through map once and sort each list
    for (auto const &pair : vecmap)
    {
        std::string categoryName = pair.first;
        std::vector<std::string> vpair = pair.second;
        std::sort(vpair.begin(), vpair.end());

        double firstNum = -1;
        double secondNum = -1;

        for (std::string x : vpair)
        {
            if (firstNum == -1 && secondNum == -1)
            {
                firstNum = atof(x.c_str());
            }
            else if (firstNum != -1 && secondNum == -1)
            {
                secondNum = atof(x.c_str());
            }
        }
        double sum = firstNum + secondNum;
        if (sum < smallest)
        {
            smallest = sum;
            smallestCategory = categoryName;
        }
    }
    return smallestCategory;
}


//
//  filter.h
//  Created by Hongchao Fang on 10/22/23.
//

// the distance calulation, dialation, erosion, cluster to support vidDisplayCPP.project 3 header file



#ifndef __filter_h
#define __filter_h

#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
cv::Mat thresholded(cv::Mat& src);
cv::Mat morphological(cv::Mat& src);
int computeM(cv::Mat& src, float alpha);
std::vector<float> compute_features(cv::Mat& src, cv::Mat& dst);
std::string findObject(std::vector<std::tuple<std::string, std::vector<float>>> db, std::vector<float> features);
std::string kNearestNeighbor(int k, std::vector<std::tuple<std::string, std::vector<float>>> db, std::vector<float> features);
#endif#pragma once

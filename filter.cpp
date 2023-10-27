//
//  filter.cpp
//  Created by Hongchao Fang on 10/22/23.
//

//the distance calulation , dialation, erosion, cluster to support vidDisplayCPP.


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core/core.hpp"
#include <numeric>
#include <map>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <filter.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
using namespace cv;


//apply thresholded to the img, any thing brigher than sum 150 will be white
cv::Mat thresholded(cv::Mat& src) {
    cv::Mat dst;
    dst.create(src.size(), src.type());
    int n = src.rows;
    int m = src.cols;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cv::Vec3b p = src.at<cv::Vec3b>(i, j);
            cv::Vec3b r = dst.at<cv::Vec3b>(i, j);

            if ((p[0] + p[1] + p[2]) < 350) {
                dst.at<cv::Vec3b>(i, j) = { 255, 255, 255 };
            }
            else {
                dst.at<cv::Vec3b>(i, j) = { 0, 0, 0 };
            }
        }
    }
    return dst;

}



//4 neighbor dialation
cv::Mat dialation(cv::Mat& src) {
    cv::Mat dst = src.clone();
    int n = src.rows;
    int m = src.cols;
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < m - 1; j++) {
            cv::Vec3b r_t = src.at<cv::Vec3b>(i + 1, j);
            cv::Vec3b r_b = src.at<cv::Vec3b>(i - 1, j);
            cv::Vec3b r_l = src.at<cv::Vec3b>(i, j - 1);
            cv::Vec3b r_r = src.at<cv::Vec3b>(i, j + 1);
            if ((r_t[0] + r_b[0] + r_l[0] + r_r[0]) > 0) {
                dst.at<cv::Vec3b>(i, j) = { 255, 255, 255 };
            }
        }
    }
    return dst;
}

//8 neighbor erosion
cv::Mat erosion(cv::Mat& src) {
    cv::Mat dst = src.clone();
    int n = src.rows;
    int m = src.cols;
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < m - 1; j++) {
            cv::Vec3b r_t = src.at<cv::Vec3b>(i + 1, j);
            cv::Vec3b r_b = src.at<cv::Vec3b>(i - 1, j);
            cv::Vec3b r_l = src.at<cv::Vec3b>(i, j - 1);
            cv::Vec3b r_r = src.at<cv::Vec3b>(i, j + 1);
            cv::Vec3b r_tl = src.at<cv::Vec3b>(i + 1, j - 1);
            cv::Vec3b r_tr = src.at<cv::Vec3b>(i + 1, j + 1);
            cv::Vec3b r_bl = src.at<cv::Vec3b>(i - 1, j - 1);
            cv::Vec3b r_br = src.at<cv::Vec3b>(i - 1, j + 1);
            if ((r_t[0] * r_b[0] * r_l[0] * r_r[0] * r_tl[0] * r_tr[0] * r_bl[0] * r_br[0]) == 0) {
                dst.at<cv::Vec3b>(i, j) = { 0, 0, 0 };
            }
        }
    }
    return dst;
}

//do morphological
cv::Mat morphological(cv::Mat& src) {

    for (int i = 0; i < 7; i++) {
        src = dialation(src);

    }
    for (int i = 0; i < 7; i++) {
        src = erosion(src);

    }
    return src;

}

//find the euclideanDist
float euclideanDist(Point& p, Point& q) {
    Point diff = p - q;
    return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}


//find the outter box and u of the img and  features {max(x_max, abs(x_min))/count, max(y_max, abs(y_min)) / count, u30 / count, u03 / count, u20 / count, u02 / count, u11 / count, percentage}s
std::vector<float> compute_outter_box(cv::Mat& regionMap, cv::Mat& dst, int i_bar, int j_bar, float alpha) {

    float x_max = -regionMap.cols;
    float y_max = -regionMap.rows;
    float u20 = 0;
    float u02 = 0;
    float u30 = 0;
    float u03 = 0;
    float u11 = 0;
    float x_min = regionMap.cols;
    float y_min = regionMap.rows;
    float u22 = 0;
    int count = 0;
    for (int i = 0; i < regionMap.rows; i++)
    {
        for (int j = 0; j < regionMap.cols; j++)
        {
            int k = regionMap.at<int>(i, j);
            if (k == 1) {
                count += 1;
                float xp = (j - j_bar) * sin(alpha) + (i - i_bar) * cos(alpha); // minor axes
                float yp = (j - j_bar) * cos(alpha) + (i - i_bar) * (-1 * sin(alpha));  // major axes

                x_max = max(x_max, xp);
                x_min = min(x_min, xp);

                y_max = max(y_max, yp);
                y_min = min(y_min, yp);
                u02 += yp * yp;
                u20 += xp * xp;
                u11 += xp * yp;
                u30 += xp * xp * xp;
                u03 += yp * yp * yp;
            }
        }
    }
    int c1_i = (x_max * cos(alpha)) + (y_max * sin(alpha));
    int c1_j = (x_max * sin(alpha)) - (y_max * cos(alpha));

    int c2_i = (x_max * cos(alpha)) + (y_min * sin(alpha));
    int c2_j = (x_max * sin(alpha)) - (y_min * cos(alpha));

    int c4_i = (x_min * cos(alpha)) + (y_max * sin(alpha));
    int c4_j = (x_min * sin(alpha)) - (y_max * cos(alpha));


    int c3_i = (x_min * cos(alpha)) + (y_min * sin(alpha));
    int c3_j = (x_min * sin(alpha)) - (y_min * cos(alpha));


    cv::Point c1 = cv::Point(c1_j + j_bar, c1_i + i_bar);
    cv::Point c2 = cv::Point(c2_j + j_bar, c2_i + i_bar);
    cv::Point c3 = cv::Point(c3_j + j_bar, c3_i + i_bar);

    cv::Point c4 = cv::Point(c4_j + j_bar, c4_i + i_bar);


    cv::circle(dst, c1, 7, Scalar(255, 255, 255));
    cv::circle(dst, c2, 7, Scalar(255, 255, 255));
    cv::circle(dst, c3, 7, Scalar(255, 255, 255));

    cv::circle(dst, c4, 7, Scalar(255, 255, 255));

    cv::line(dst, c1, c2, cv::Scalar(255, 0, 0), 5);
    cv::line(dst, c2, c3, cv::Scalar(255, 0, 0), 5);
    cv::line(dst, c4, c3, cv::Scalar(255, 0, 0), 5);
    cv::line(dst, c4, c1, cv::Scalar(255, 0, 0), 5);


    float percentage = 1.0 * count / euclideanDist(c1, c2) * euclideanDist(c2, c3);
    std::vector<float> result = {max(x_max, abs(x_min))/count, max(y_max, abs(y_min)) / count, u30 / count, u03 / count, u20 / count, u02 / count, u11 / count, percentage};


    return result;


}

//compute features {max(x_max, abs(x_min))/count, max(y_max, abs(y_min)) / count, u30 / count, u03 / count, u20 / count, u02 / count, u11 / count, percentage} and load box and axis into img
std::vector<float> compute_features(cv::Mat &src, cv::Mat &dst){
    
    dst.create(src.size(), src.type());
    cv::Mat regionMap;
    cv::Mat1i uselss;
    cv::Mat1d centroids;
    Mat temp;
    cv::cvtColor(src, temp, cv::COLOR_BGR2GRAY);
    cv::connectedComponentsWithStats(temp, regionMap, uselss, centroids, 4, CV_32S, cv::CCL_DEFAULT);
    int top = src.rows;
    int left = src.cols;
    int right = 0;
    int bot = 0;
    int M11 = 0;
    int M20 = 0;
    int M02 = 0;
    int j_bar = centroids(1, 0);
    int i_bar = centroids(1, 1);
    int count = 0;
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            int k = regionMap.at<int>(i, j);
            if (k == 1) {
                dst.at<cv::Vec3b>(i, j) = { 150, 0, 150 };
                M11 += (i - i_bar) * (j - j_bar);
                M20 += (i - i_bar) * (i - i_bar);
                M02 += (j - j_bar) * (j - j_bar);
                count += 1;
            }
            else if (k == 2)
            {
                dst.at<cv::Vec3b>(i, j) = { 0, 150, 0 };
            }
            else if (k == 3)
            {
                dst.at<cv::Vec3b>(i, j) = { 0, 0, 150 };
            }
            else if (k == 4)
            {
                dst.at<cv::Vec3b>(i, j) = { 150, 0, 0 };
            }
        }
    }
    if (count < 30) {
        return {};
    }
    float alpha = 0.5 * atan2(2 * M11, M20 - M02);
    // cv::Point point1 = cv::Point(j_bar, i_bar);
    // cv::Point point2 = cv::Point(j_bar + 100 * sin(alpha), i_bar + 100 * cos(alpha));
    // cv::circle(dst, point1, 7, Scalar(0, 255, 0));
    // cv::line(dst, point1, point2, Scalar(0, 255, 0), 3);

    return compute_outter_box(regionMap, dst, i_bar, j_bar, alpha);

}

//find stddev of each vector
double stddev(std::vector<double> const& func)
{
    double mean = std::accumulate(func.begin(), func.end(), 0.0) / func.size();
    double sq_sum = std::inner_product(func.begin(), func.end(), func.begin(), 0.0,
        [](double const& x, double const& y) { return x + y; },
        [mean](double const& x, double const& y) { return (x - mean) * (y - mean); });
    return std::sqrt(sq_sum / func.size());
}

//find the sdtv of each input in db.
std::vector<float> findSdtV(std::vector<std::tuple<std::string, std::vector<float>>> db) {
    std::vector<float> std_v;
    int n = std::get<1>(db.at(0)).size();
    std::vector<std::vector<float>> db_f;
    for (int i = 0; i < n; i++) {
        std::vector<float> temp;
        db_f.push_back(temp);
    }

    for (std::tuple<std::string, std::vector<float>> t : db) {
        for (int i = 0; i < n; i++) {

            float f = std::get<1>(t)[i];
            db_f.at(i).push_back(f);
        }
    }
    //cr: https://stackoverflow.com/questions/7616511/calculate-mean-and-standard-deviation-from-a-vector-of-samples-in-c-using-boos
    for (std::vector<float> v : db_f) {
        float sum = std::accumulate(v.begin(), v.end(), 0.0);
        float mean = sum / v.size();
        float sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
        float stdev = std::sqrt(sq_sum / v.size() - mean * mean);
        std_v.push_back(stdev);
    }
    return std_v;
}

//Manhattan distance for extension (it is bad!!!)
float distanceManhattan(std::vector<float> db, std::vector<float> features, std::vector<float> std_v, int n) {
    float diff = 0;
    for (int i = 0; i < n; i++) {
        float x1 = features[i];
        float x2 = db[i];
        diff += abs(x1 - x2);
    }
    return diff;
}

//Minkowski distance for extension (also bad since it not normalized)
float distanceMinkowski(std::vector<float> db, std::vector<float> features, std::vector<float> std_v, int n) {
    float diff = 0;
    for (int i = 0; i < n; i++) {
        float x1 = features[i];
        float x2 = db[i];
        diff += (abs(x1 - x2) * abs(x1 - x2));
    }
    return sqrt(diff);
}

//Eucliden distance given in canvas
float distanceEuclidean(std::vector<float> db, std::vector<float> features, std::vector<float> std_v, int n) {
    float diff = 0;
    for (int i = 0; i < n; i++) {
        float x1 = features[i];
        float x2 = db[i];
        diff += (abs(x1 - x2) / std_v[i]);
    }
    return diff;
}

//find the closet object by using Euclidean distance metric with nearest-neighbor recognition
std::string findObject(std::vector<std::tuple<std::string, std::vector<float>>> db, std::vector<float> features) {
    if (db.empty() || features.empty()) {
        return "EMPTY";
    }
    int n = std::get<1>(db.at(0)).size();
    std::vector<float> std_v = findSdtV(db);
    std::string result;
    float min_diff = -1;
    for (std::tuple<std::string, std::vector<float>> t : db) {
        std::string name = std::get<0>(t);
        float diff = distanceEuclidean(std::get<1>(t), features, std_v, n);
        if (min_diff == -1) {
            min_diff = abs(diff);
            result = name;
        }
        else if (min_diff > abs(diff)) {
            result = name;
            min_diff = abs(diff);
        }
    }
    //handle unknown
    if (min_diff >= 3) {
        return "UNKNOWN";
    }
    return result;

}
//sort given the first index of input
bool sortbyfirst(const std::tuple<int, String>& a, const std::tuple<int, String>& b) {
    return (std::get<0>(a) < std::get<0>(b));
}

//sort the result from different matchingmethod and return number of min difference path given numImages
std::vector<std::string> sortHelper(std::vector<std::tuple <float, std::string>> input, int k) {
    std::vector<std::string> result;
    sort(input.begin(), input.end(), sortbyfirst);
    for (int i = 0; i < input.size() && i <= k; i++) {
        result.push_back(std::get<1>(input.at(i)));
    }
    //handel unknown
    if (std::get<0>(input.at(0)) >= 3) {
        return { "UNKNOWN", "UNKNOWN" };
    }
    return result;
}


std::string kNearestNeighbor(int k, std::vector<std::tuple<std::string, std::vector<float>>> db, std::vector<float> features)
{   
    if (db.empty() || features.empty()) {
        return "EMPTY";
    }
    std::map<std::string, int> count;
    std::vector<float> std_v = findSdtV(db);
    int n = std::get<1>(db.at(0)).size();
    std::vector<std::tuple <float, std::string>> result;
    for (std::tuple<std::string, std::vector<float>> t : db) {
        std::string name = std::get<0>(t);
        float diff = distanceEuclidean(std::get<1>(t), features, std_v, n);
        result.push_back(std::make_tuple(diff, name));
    }
    std::vector<std::string> r = sortHelper(result, k);
    int m_val = 0;
    std::string rt;
    for (std::string i : r) {
        if (count.count(i) == 0) {
            count.insert({ i, 0 });
        }
        count[i] += 1;
        if (count[i] > m_val) {
            m_val = count[i];
            rt = i;
        }
    }
    return rt;
}
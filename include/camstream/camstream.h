#pragma once

#include <unistd.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "camstream/camstream.h"
#include "yolo/cpm.hpp"
#include "yolo/infer.hpp"
#include "yolo/yolo.hpp"
#include <vector>

class streamer
{
public:

    streamer();

    ~streamer();

    int light(cv::Mat img) const;

    bool blurry_det(cv::Mat img, double threshold);

    void camStatusDet(cv::Mat frame, int *over_exp_r, int *block_r, int *blur_r);

    char *run_frame(std::vector<cv::Mat> img_batch);

    void put_char_front_back(int b, std::vector<yolo::BoxArray> &batched_result, std::vector<cv::Mat> &img_batch);

    void put_char_right_left(int b, std::vector<yolo::BoxArray> &batched_result, std::vector<cv::Mat> &img_batch);

private:
    // Configuration related parameters
    std::vector<int> RoI;
    cv::Rect r_l, r_r;
    cv::Mat mask_f, mask_b, out;
    std::vector<cv::Point> contour_f, contour_b;
    std::vector<std::vector<cv::Point>> contours_f, contours_b;

    // Image quality detection parameters
    int exp_region, block_region, blur_region;
    const static int blurry_thre = 100;
    const static int darkness_thre = 60;
    const static int over_exp_thre = 230;

    // Detection result storage for each camera
    char obj_f[2];
    char obj_b[2];
    char obj_l[2];
    char obj_r[2];
    char out_char[8] = {'0', '0', '0', '0', '0', '0', '0', '0'};

    char detection_result[10];  // Buffer for storing detection results
};


#include "camstream/camstream.h"
#include "yaml_cpp/yaml.h"

using namespace std;
using namespace cv;

// load the detection model
string engine_path = "../workspace/best.engine";
auto yolomodel = yolo::load(engine_path, yolo::Type::V8);

yolo::Image cvimg(const cv::Mat& image) {
    return yolo::Image(image.data, image.cols, image.rows);
}

// Define a function to get rectangle region from configuration file
Rect getRectFromConfig(const YAML::Node& config, const string& camera);
void getPointsFromConfig(const YAML::Node& config, const string& camera, vector<Point>& contour);

streamer::streamer() {
    // Read configuration file for camera ROI settings
    YAML::Node config = YAML::LoadFile("../config/cam_roi.yaml");
    if (!config) {
        cout << "Open config File failed.";
        exit(-1);
    }

    // Initialize masks for front and back camera region filtering
    mask_f = Mat(360, 640, CV_8UC1, Scalar(0, 0, 0));
    mask_b = Mat(360, 640, CV_8UC1, Scalar(0, 0, 0));
    Rect r_l = getRectFromConfig(config, "left_camera");
    Rect r_r = getRectFromConfig(config, "right_camera");

    // Get contour points from configuration for front and back cameras
    getPointsFromConfig(config, "front_camera", contour_f);
    getPointsFromConfig(config, "back_camera", contour_b);
    contours_f.emplace_back(contour_f);
    contours_b.emplace_back(contour_b);
    // Draw filled contours on masks for region of interest filtering
    drawContours(mask_f, contours_f, 0, 1, FILLED);
    drawContours(mask_b, contours_b, 0, 1, FILLED);
}

// Use default destructor
streamer::~streamer() = default;

char* streamer::run_frame(vector<Mat> img_batch) {
    // Initialize output character array to all zeros
    for (int ii = 0; ii < 8; ++ii) {
        out_char[ii] = '0';
    }

    // Convert OpenCV Mat images to YOLO format for batch inference
    vector<yolo::Image> yoloimages(img_batch.size());
    transform(img_batch.begin(), img_batch.end(), yoloimages.begin(), cvimg);
    auto batched_result = yolomodel->forwards(yoloimages);
    
    // Process detection results for each camera position
    put_char_front_back(0, batched_result, img_batch);  // Front camera
    put_char_front_back(1, batched_result, img_batch);  // Back camera
    put_char_right_left(2, batched_result, img_batch);  // Left camera
    put_char_right_left(3, batched_result, img_batch);  // Right camera
    
    /* Output inference results */
    // cout<<"out_char: "<<out_char<<endl;
    // out_char: 00000001
    return out_char;
}

// Front and back cameras use irregular shapes, requiring drawContours + point extraction for IOU calculation
void streamer::put_char_front_back(int b, vector<yolo::BoxArray>& batched_result, vector<Mat>& img_batch) {
    auto& objs = batched_result[b];
    auto& image = img_batch[b];
    for (auto& obj : objs) {
        // Draw detection bounding box with random color
        uint8_t cb, cg, cr;
        tie(cb, cg, cr) = yolo::random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(cb, cg, cr), 5);

        // Create contour from bounding box for intersection calculation
        vector<Point> contour_t;
        vector<vector<Point>> contours_t;
        contour_t.emplace_back(Point2f(double(obj.left), double(obj.top)));
        contour_t.emplace_back(Point2f(double(obj.left), double(obj.bottom)));
        contour_t.emplace_back(Point2f(double(obj.right), double(obj.bottom)));
        contour_t.emplace_back(Point2f(double(obj.right), double(obj.top)));
        contours_t.emplace_back(contour_t);
        Mat mask_tmp(360, 640, CV_8UC1, Scalar(0, 0, 0));
        drawContours(mask_tmp, contours_t, 0, 1, FILLED);
        bitwise_and(mask_f, mask_tmp, out);

        // Return count of non-zero (white) pixels - indicates intersection with ROI
        if (countNonZero(out) > 0) {
            if (((obj.class_label) == 1) && b == 0) {
                // cout << "front cam: human detected!" << endl;
                out_char[0] = {'1'};  // Front camera human detection
            } else if (((obj.class_label) != 1) && b == 0) {
                out_char[1] = {'1'};  // Front camera other objects detection
            } else if (((obj.class_label) == 1) && b == 1) {
                // cout << "back cam: human detected!" << endl;
                out_char[2] = {'1'};  // Back camera human detection
            }
        }
    }
}

void streamer::put_char_right_left(int b, vector<yolo::BoxArray>& batched_result, vector<Mat>& img_batch) {
    auto& objs = batched_result[b];
    auto& image = img_batch[b];
    for (auto& obj : objs) {
        // Draw detection bounding box with random color
        uint8_t cb, cg, cr;
        tie(cb, cg, cr) = yolo::random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(cb, cg, cr), 5);
        Rect r(cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom));
        
        // Check intersection with right/left camera ROI rectangle
        Rect r_tmp = r & r_r;
        if (r_tmp.area() > 0) {
            if (((obj.class_label) == 1) && b == 2) {  // class 1 for human, not same with coco
                out_char[4] = {'1'};  // Left camera human detection
            } else if (((obj.class_label) != 1) && b == 2) {
                out_char[5] = {'1'};  // Left camera obstacles detection
            } else if (((obj.class_label) == 1) && b == 3) {
                out_char[6] = {'1'};  // Right camera human detection
            } else if (((obj.class_label) != 1) && b == 3) {
                out_char[7] = {'1'};  // Right camera obstacles detection
            }
        }
    }
}

/*
If an image is particularly bright, its average value will increase.
This is because bright pixels have higher values than dark pixels, 
so they contribute more to the average value.
*/
int streamer::light(Mat img) const {
    Scalar scalar = mean(img);
    return scalar.val[0];
}

/*
In images, edges and details usually have higher contrast, while blurred areas do not.
Therefore, standard deviation can be used to measure image sharpness.
If the standard deviation is small, the image may be blurred; 
if the standard deviation is large, the image may be sharp.
*/
// Blur detection using Laplacian variance method
bool streamer::blurry_det(Mat img, double threshold) {
    Mat matImageGray(img.rows, img.cols, CV_8UC1, Scalar(0));
    cvtColor(img, matImageGray, COLOR_BGR2GRAY);
    Mat dst, abs_dst;
    // Additionally, Laplacian transform may produce negative values,
    // so the target image data type needs to be set to an appropriate range (like CV_16S) to avoid data overflow.
    Laplacian(matImageGray, dst, CV_16S);

    // Convert to 8-bit unsigned integer (absolute value)
    convertScaleAbs(dst, abs_dst);
    Mat tmp_m, tmp_sd;
    double sd = 0;
    meanStdDev(dst, tmp_m, tmp_sd);
    sd = tmp_sd.at<double>(0, 0);
    return ((sd * sd) <= threshold);
}

// Soil cover & over-exposure detection
void streamer::camStatusDet(Mat frame, int* over_exp_r, int* block_r, int* blur_r) {
    // Divide frame into 4 quadrants for comprehensive analysis
    Mat frame_ul, frame_ur, frame_bl, frame_br;
    Rect rect_ul(0, 0, frame.cols / 2, frame.rows / 2);  // Upper left
    frame_ul = Mat(frame, rect_ul);
    Rect rect_ur(frame.cols / 2, 0, frame.cols / 2, frame.rows / 2);  // Upper right
    frame_ur = Mat(frame, rect_ur);
    Rect rect_bl(0, frame.rows / 2, frame.cols / 2, frame.rows / 2);  // Bottom left
    frame_bl = Mat(frame, rect_bl);
    Rect rect_br(frame.cols / 2, frame.rows / 2, frame.cols / 2, frame.rows / 2);  // Bottom right
    frame_br = Mat(frame, rect_br);

    // Calculate brightness levels for each quadrant
    double ll_ul = light(frame_ul);
    double ll_ur = light(frame_ur);
    double ll_bl = light(frame_bl);
    double ll_br = light(frame_br);
    
    // Count quadrants with different quality issues
    // darkness_thre is 60; over_exp_thre is 230;
    int blur_n = blurry_det(frame_ul, blurry_thre) + blurry_det(frame_ur, blurry_thre) +
                 blurry_det(frame_bl, blurry_thre) + blurry_det(frame_br, blurry_thre);
    int block_n = (ll_ul < darkness_thre) + (ll_ur < darkness_thre) + (ll_bl < darkness_thre) + (ll_br < darkness_thre);
    int overexposure_n = (ll_ul > over_exp_thre) + (ll_ur > over_exp_thre) + (ll_bl > over_exp_thre) + (ll_br > over_exp_thre);
    
    // Return analysis results
    *over_exp_r = overexposure_n;
    *blur_r = blur_n;
    *block_r = block_n;
}

// Define a function to get rectangle region from configuration file
Rect getRectFromConfig(const YAML::Node& config, const string& camera) {
    // Try to get corresponding values
    try {
        int x = config[camera]["x"].as<int>();
        int y = config[camera]["y"].as<int>();
        int width = config[camera]["width"].as<int>();
        int height = config[camera]["height"].as<int>();
        // Return a rectangle object
        return Rect(x, y, width, height);
    } catch (const YAML::Exception& e) {
        // If an exception occurs, output error message and return an empty rectangle
        cerr << "Error: " << e.what() << endl;
        return Rect();
    }
}

// Define a function to get four points from configuration file and add to contour
void getPointsFromConfig(const YAML::Node& config, const string& camera, vector<Point>& contour) {
    // Try to get corresponding values
    try {
        // Get coordinates of four points
        vector<int> point1 = config[camera]["point1"].as<vector<int>>();
        vector<int> point2 = config[camera]["point2"].as<vector<int>>();
        vector<int> point3 = config[camera]["point3"].as<vector<int>>();
        vector<int> point4 = config[camera]["point4"].as<vector<int>>();
        // Add four points to contour
        contour.emplace_back(Point2f(point1[0], point1[1]));
        contour.emplace_back(Point2f(point2[0], point2[1]));
        contour.emplace_back(Point2f(point3[0], point3[1]));
        contour.emplace_back(Point2f(point4[0], point4[1]));
    } catch (const YAML::Exception& e) {
        // If an exception occurs, output error message
        cerr << "Error: " << e.what() << endl;
    }
}
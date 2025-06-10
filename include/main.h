#include <objects.pb.h>
#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using namespace std;
using namespace cv;
using namespace phoenix;

// Global image matrices for 4 cameras
Mat img[4];
Mat img_tmp[4];
Mat img_vis[4], img_o[4];
VideoCapture capture[4];

// Camera device paths for USB video capture devices
string cam_path[4]={"/dev/v4l/by-path/platform-3530000.xhci-usb-0:3.2:1.0-video-index0", 
                         "/dev/v4l/by-path/platform-3530000.xhci-usb-0:3.1:1.0-video-index0", 
                         "/dev/v4l/by-path/platform-3530000.xhci-usb-0:2.1:1.0-video-index0", 
                         "/dev/v4l/by-path/platform-3530000.xhci-usb-0:2.2:1.0-video-index0"};

// Radar visualization image matrices
Mat img_vis_radar;
Mat radar_img = imread("coordinate.png");
Mat out_radar = radar_img(Rect(0, 3*radar_img.rows/4, radar_img.cols , 3*radar_img.rows/16));
char* cam_det_res;

// Radar coordinate transformation parameters
float circle_x;
float circle_y;
const float fx = 20.0;  // X-axis scaling factor for radar coordinate conversion
const float fy = 6.5;   // Y-axis scaling factor for radar coordinate conversion
const int zero = 975;   // Zero reference point for radar display
const double MAX_DISTANCE = 15; // Maximum detection distance in meters
const double SCALE_FACTOR = 100; // Scaling factor for coordinate transformation
const int CIRCLE_RADIUS = 4; // Circle radius for object visualization

// Radar feature structure definition
struct feature
{
    int id;
    float distance_x;
    float distance_y;
    float vlong;
    float vlat;
    float angle;
};

// Utility function to get current timestamp as string
inline string getTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%m%d_%H%M%S", localtime(&timep));
    return tmp;
}

// Object tracking and filtering variables
map<int, int> continuous_ids_;  // Track continuous detection frames for each object ID
vector<feature> information_list_new;  // New radar detection information list

int delay_frames_=5;  // Minimum frames required before accepting an object detection
int exp_region = 0;   // Over-exposure region count
int block_region = 0; // Blocked/dark region count  
int blur_region = 0;  // Blurred region count

int cam_others_flag, cam_human_flag;  // Camera detection flags

// Communication channel identifiers, now used only for printing
bool publisher_f_enabled = true;
bool publisher_b_enabled = true;
bool publisher_l_enabled = true;
bool publisher_r_enabled = true;
bool publisher_MM_enabled = true;
bool publisher_UI_enabled = true;
bool publisher_obj_enabled = true;
bool publisher_obj_c_enabled = true;

// Streaming configuration variables
int hb_seq, tractor_id, snapshot_position;
string cam_url[4];
int stream_time[4], cam_time[4];
VideoWriter m_video_output[4];
string gstreamer_pipeline_out[4];
Mat m_frame = Mat::eye(360, 640, CV_8UC3);

// Network configuration constants
#define SERVER_PORT 6669
#define BUFFER_SIZE 1024
char http_ip[100];
char m_uploadLog_dir[100];
char m_uploadLog_fileName[100];
static const char* m_achCWD = "/home/phoenix";
static const char IP_address[] = "192.168.0.100"; // Server IP address
int http_port;

// Snapshot file paths for different camera positions and radar
string snapshot_path[5]={"/home/phoenix/front.jpg", "/home/phoenix/back.jpg", 
                              "/home/phoenix/left.jpg", "/home/phoenix/right.jpg",
                              "/home/phoenix/radar.jpg"};

// Convert camera position to internal camera ID
static inline int position2id(int pos)
{
    int cam_id_func;
    static const int cam_id_table[] = {0, 1, 4, 2, 3};  // Position mapping table
    cam_id_func = cam_id_table[pos];
    return cam_id_func;
}

// Compile-time hash function for string constants
template <typename _T>
unsigned int constexpr Hash(_T const* input) {
    return *input ? static_cast<unsigned int>(*input) + 33 * Hash(input + 1) : 5381;
}

// Simple Base64 encoding functionality implementation
class SimpleBase64 {
public:
    static std::string encode(const std::string& input);
    static std::string encode_file(const std::string& filename);
};

// Simple logging macro definitions to replace external logging system
#define LOGINFO(level, format, ...) \
    do { \
        printf("[INFO] " format "\n", ##__VA_ARGS__); \
        fflush(stdout); \
    } while(0)

#define InitLog() do {} while(0)
#define IDC_SetLoggerIP(ip, port) do {} while(0)
#define FL_Init(a, b, c) do {} while(0)

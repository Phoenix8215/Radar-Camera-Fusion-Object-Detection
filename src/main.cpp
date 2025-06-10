#include <dirent.h>
#include <fcntl.h>
#include <json/json.h>
#include <main.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>
#include <chrono>
#include <memory>
#include <optional>
#include <string_view>
#include <filesystem>
#include <algorithm>
#include <execution>
#include <variant>

#include "camstream/camstream.h"
#include "radar/radar.h"

using namespace std::chrono_literals;
namespace fs = std::filesystem;

// Thread-safe globals using modern C++ primitives
inline std::atomic<bool> g_shutdown_requested{false};
inline std::atomic<int> g_close_count{0};
inline std::atomic<int> g_hb_seq{0};
inline std::atomic<int> g_tractor_id{0};

// Thread synchronization
inline std::mutex g_img_mutex;
inline std::mutex g_radar_mutex;
inline std::mutex g_log_mutex;
inline std::condition_variable g_img_cv;

// Modern replacements for global variables
inline std::array<std::atomic<time_t>, 4> g_cam_time{};
inline std::array<std::atomic<time_t>, 4> g_stream_time{};
inline std::array<cv::Mat, 4> g_img_safe; // Protected by g_img_mutex
inline cv::Mat g_out_radar_safe; // Protected by g_radar_mutex

// Upload log state
struct UploadLogState {
    std::string directory;
    std::string filename;
    std::atomic<bool> is_uploading{false};
};
inline UploadLogState g_upload_log_state;

// Base64 encoding implementation
class SimpleBase64 {
public:
    static std::string encode(std::string_view input) {
        static constexpr std::string_view chars = 
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        
        std::string result;
        result.reserve(((input.size() + 2) / 3) * 4);
        
        int val = 0, valb = -6;
        for (unsigned char c : input) {
            val = (val << 8) + c;
            valb += 8;
            while (valb >= 0) {
                result.push_back(chars[(val >> valb) & 0x3F]);
                valb -= 6;
            }
        }
        if (valb > -6) result.push_back(chars[((val << 8) >> (valb + 8)) & 0x3F]);
        while (result.size() % 4) result.push_back('=');
        return result;
    }

    static std::optional<std::string> encode_file(const fs::path& filename) {
        if (!fs::exists(filename)) return std::nullopt;
        
        std::ifstream file(filename, std::ios::binary);
        if (!file) return std::nullopt;
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        return encode(content);
    }
};

// Modern RAII socket wrapper
class SocketWrapper {
private:
    int m_socket;
    
public:
    explicit SocketWrapper(int domain, int type, int protocol = 0) 
        : m_socket(socket(domain, type, protocol)) {
        if (m_socket < 0) {
            throw std::runtime_error("Socket creation failed: " + std::string(strerror(errno)));
        }
    }
    
    ~SocketWrapper() {
        if (m_socket >= 0) {
            close(m_socket);
        }
    }
    
    // Non-copyable but movable
    SocketWrapper(const SocketWrapper&) = delete;
    SocketWrapper& operator=(const SocketWrapper&) = delete;
    
    SocketWrapper(SocketWrapper&& other) noexcept : m_socket(other.m_socket) {
        other.m_socket = -1;
    }
    
    SocketWrapper& operator=(SocketWrapper&& other) noexcept {
        if (this != &other) {
            if (m_socket >= 0) close(m_socket);
            m_socket = other.m_socket;
            other.m_socket = -1;
        }
        return *this;
    }
    
    int get() const noexcept { return m_socket; }
    
    void bind_to(const sockaddr* addr, socklen_t addrlen) {
        if (bind(m_socket, addr, addrlen) < 0) {
            throw std::runtime_error("Bind failed: " + std::string(strerror(errno)));
        }
    }
    
    void connect_to(const sockaddr* addr, socklen_t addrlen) {
        if (connect(m_socket, addr, addrlen) < 0) {
            throw std::runtime_error("Connect failed: " + std::string(strerror(errno)));
        }
    }
    
    ssize_t read_data(void* buf, size_t count) {
        return read(m_socket, buf, count);
    }
    
    ssize_t write_data(const void* buf, size_t count) {
        return write(m_socket, buf, count);
    }
};

// Modern thread-safe logger
class ThreadSafeLogger {
private:
    mutable std::mutex m_mutex;
    
public:
    template<typename... Args>
    void log_info(std::string_view format, Args&&... args) {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Use std::format in C++20, for now use printf-style
        printf(format.data(), std::forward<Args>(args)...);
        printf("\n");
    }
};

inline ThreadSafeLogger g_logger;

// Global objects - using smart pointers for better resource management
inline std::unique_ptr<SimpleBase64> g_base64_encoder = std::make_unique<SimpleBase64>();
inline std::unique_ptr<Radar_SF73> g_radar = std::make_unique<Radar_SF73>();
inline std::unique_ptr<streamer> g_cam_stream = std::make_unique<streamer>();

// Thread-safe object creation helper
template<typename T>
void create_object_from_radar_point(T& object, cv::Point2f radar_p, int id, double vlong, double vlat) {
    object->set_label("2");
    object->set_confidence(0.7);
    object->set_x_dist(radar_p.y);
    object->set_y_dist(-radar_p.x);
    object->set_x_vel(vlong);
    object->set_y_vel(vlat);
    object->set_theta(0);
    object->set_width(0.2);
    object->set_length(0.2);
    object->set_id(id);
}

// Draw circle with thread safety
void draw_circle_from_object(const Object* object, cv::Mat& img_vis_radar) {
    constexpr double SCALE_FACTOR = 320.0; // Example value
    constexpr double fx = 10.0, fy = 10.0; // Example values
    constexpr double zero = 180.0; // Example value
    constexpr int CIRCLE_RADIUS = 5;
    
    double circle_x = SCALE_FACTOR + fx * (-object->y_dist());
    double circle_y = zero - fy * (object->x_dist());
    cv::circle(img_vis_radar, cv::Point(circle_x, circle_y), CIRCLE_RADIUS,
               cv::Scalar(0, 0, 255), -1, 8);
}

class CameraAnalyzer {
private:
    std::atomic<bool> m_running{true};
    std::thread m_thread;
    
public:
    CameraAnalyzer() : m_thread(&CameraAnalyzer::analyze_loop, this) {}
    
    ~CameraAnalyzer() {
        stop();
    }
    
    void stop() {
        m_running = false;
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }
    
private:
    void analyze_loop() {
        g_logger.log_info("Camera analysis thread started");
        
        while (m_running && !g_shutdown_requested) {
            // Use parallel execution policy for processing multiple cameras
            std::vector<int> camera_indices{0, 1, 2, 3};
            
            std::for_each(std::execution::par_unseq, camera_indices.begin(), camera_indices.end(),
                [this](int i) {
                    std::unique_lock<std::mutex> lock(g_img_mutex);
                    if (!g_img_safe[i].empty()) {
                        lock.unlock(); // Release lock before processing
                        
                        int exp_region = 0, block_region = 0, blur_region = 0;
                        g_cam_stream->camStatusDet(g_img_safe[i], &exp_region, &block_region, &blur_region);
                        
                        g_logger.log_info("Camera %d - exp_region: %d, block_region: %d, blur_region: %d",
                                        i, exp_region, block_region, blur_region);
                    }
                });
            
            std::this_thread::sleep_for(1s);
        }
        
        g_logger.log_info("Camera analysis thread stopped");
    }
};

// Radar receiver thread
class RadarReceiver {
private:
    std::atomic<bool> m_running{true};
    std::thread m_thread;
    
public:
    RadarReceiver() : m_thread(&RadarReceiver::receive_loop, this) {}
    
    ~RadarReceiver() {
        stop();
    }
    
    void stop() {
        m_running = false;
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }
    
private:
    void receive_loop() {
        g_logger.log_info("Radar receiver thread started");
        
        while (m_running && !g_shutdown_requested) {
            g_radar->receive_data();
            std::this_thread::sleep_for(10ms);
        }
        
        g_logger.log_info("Radar receiver thread stopped");
    }
};

class RadarProcessor {
private:
    std::atomic<bool> m_running{true};
    std::thread m_thread;
    std::unordered_map<int, int> m_continuous_ids;
    mutable std::mutex m_data_mutex;
    
    static constexpr int DELAY_FRAMES = 5;
    static constexpr double MAX_DISTANCE = 100.0;
    
public:
    RadarProcessor() : m_thread(&RadarProcessor::process_loop, this) {}
    
    ~RadarProcessor() {
        stop();
    }
    
    void stop() {
        m_running = false;
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }
    
private:
    void process_loop() {
        g_logger.log_info("Radar processor thread started");
        
        while (m_running && !g_shutdown_requested) {
            process_radar_data();
            std::this_thread::sleep_for(100ms);
        }
        
        g_logger.log_info("Radar processor thread stopped");
    }
    
    void process_radar_data() {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        
        std::unordered_map<int, int> current_con_ids;
        std::vector<feature> information_list_new;
        
        // Process radar detection data
        for (const auto& [i, radar_info] : enumerate(g_radar->information_list_pre)) {
            feature obj{
                .id = radar_info.id,
                .distance_x = radar_info.distance_x,
                .distance_y = radar_info.distance_y,
                .vlong = radar_info.vlong,
                .vlat = radar_info.vlat,
                .angle = radar_info.angle
            };
            
            // Update continuous ID tracking
            if (auto it = m_continuous_ids.find(obj.id); it != m_continuous_ids.end()) {
                current_con_ids[obj.id] = it->second + 1;
            } else {
                current_con_ids[obj.id] = 1;
            }
            
            // Filter objects based on continuous frame count
            if (current_con_ids[obj.id] > DELAY_FRAMES) {
                information_list_new.emplace_back(std::move(obj));
            }
        }
        
        m_continuous_ids = std::move(current_con_ids);
        
        if (!information_list_new.empty()) {
            process_detected_objects(information_list_new);
        }
    }
    
    void process_detected_objects(const std::vector<feature>& objects) {
        phoenix::Obj_list family, family_cal;
        family.set_ver("1.4.6");
        family_cal.set_ver("1.4.6");
        
        for (const auto& obj : objects) {
            cv::Point2f radar_p{obj.distance_x, obj.distance_y};
            
            if constexpr (true) { // Could be a compile-time condition
                if (cv::norm(radar_p) < MAX_DISTANCE) {
                    auto* object_c = family_cal.add_object();
                    create_object_from_radar_point(object_c, radar_p, obj.id, obj.vlong, obj.vlat);
                    
                    // Thread-safe drawing
                    std::lock_guard<std::mutex> radar_lock(g_radar_mutex);
                    // draw_circle_from_object(object_c, radar_img);
                }
            }
        }
        
        // Log results instead of publishing
        g_logger.log_info("【Radar Object Publisher】Published %d objects to obj_list channel", 
                         family.object_size());
        g_logger.log_info("【Radar Calculation Publisher】Published %d objects to obj_cal channel", 
                         family_cal.object_size());
    }
    
    template<typename Container>
    auto enumerate(const Container& container) {
        std::vector<std::pair<size_t, typename Container::value_type>> result;
        size_t index = 0;
        for (const auto& item : container) {
            result.emplace_back(index++, item);
        }
        return result;
    }
};

// Image capture manager using modern C++
class ImageCaptureManager {
private:
    std::atomic<bool> m_running{true};
    std::thread m_thread;
    std::array<cv::VideoCapture, 4> m_captures;
    std::array<std::string, 4> m_cam_paths;
    
public:
    explicit ImageCaptureManager(std::array<std::string, 4> cam_paths) 
        : m_cam_paths(std::move(cam_paths)), m_thread(&ImageCaptureManager::capture_loop, this) {
        
        // Initialize captures in parallel
        std::vector<int> indices{0, 1, 2, 3};
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [this](int i) {
                m_captures[i].open(m_cam_paths[i]);
                if (!m_captures[i].isOpened()) {
                    g_logger.log_info("Warning: Failed to open camera %d at path %s", 
                                    i, m_cam_paths[i].c_str());
                }
            });
    }
    
    ~ImageCaptureManager() {
        stop();
    }
    
    void stop() {
        m_running = false;
        if (m_thread.joinable()) {
            m_thread.join();
        }
        
        // Release all captures
        for (auto& capture : m_captures) {
            if (capture.isOpened()) {
                capture.release();
            }
        }
    }
    
private:
    void capture_loop() {
        g_logger.log_info("Image capture thread started");
        
        std::array<cv::Mat, 4> temp_images;
        
        while (m_running && !g_shutdown_requested) {
            // Capture from all cameras in parallel
            std::vector<int> indices{0, 1, 2, 3};
            std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                [this, &temp_images](int i) {
                    if (m_captures[i].isOpened()) {
                        cv::Mat frame;
                        m_captures[i] >> frame;
                        
                        if (!frame.empty()) {
                            g_cam_time[i] = std::time(nullptr);
                            cv::resize(frame, temp_images[i], cv::Size(640, 360), 0, 0, cv::INTER_LINEAR);
                        }
                    }
                });
            
            // Update shared images with proper synchronization
            {
                std::lock_guard<std::mutex> lock(g_img_mutex);
                for (int i = 0; i < 4; ++i) {
                    if (!temp_images[i].empty()) {
                        g_img_safe[i] = temp_images[i].clone();
                    }
                }
            }
            g_img_cv.notify_all();
            
            std::this_thread::sleep_for(30ms);
        }
        
        g_logger.log_info("Image capture thread stopped");
    }
};

// File upload manager with async operations
class FileUploadManager {
private:
    std::atomic<bool> m_running{true};
    std::thread m_thread;
    
public:
    FileUploadManager() : m_thread(&FileUploadManager::upload_loop, this) {}
    
    ~FileUploadManager() {
        stop();
    }
    
    void stop() {
        m_running = false;
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }
    
private:
    void upload_loop() {
        g_logger.log_info("File upload thread started");
        
        while (m_running && !g_shutdown_requested) {
            // Wait for images to be available
            std::unique_lock<std::mutex> lock(g_img_mutex);
            g_img_cv.wait(lock, []() {
                return !g_img_safe[0].empty() && !g_img_safe[1].empty() && 
                       !g_img_safe[2].empty() && !g_img_safe[3].empty();
            });
            
            if (g_shutdown_requested) break;
            
            // Create local copies
            std::array<cv::Mat, 4> local_images;
            for (int i = 0; i < 4; ++i) {
                local_images[i] = g_img_safe[i].clone();
            }
            lock.unlock();
            
            // Save and upload files asynchronously
            auto upload_future = std::async(std::launch::async, [this, local_images]() {
                save_and_upload_images(local_images);
            });
            
            std::this_thread::sleep_for(5s);
        }
        
        g_logger.log_info("File upload thread stopped");
    }
    
    void save_and_upload_images(const std::array<cv::Mat, 4>& images) {
        const auto now = std::chrono::system_clock::now();
        const auto time_t = std::chrono::system_clock::to_time_t(now);
        const auto timestamp = std::to_string(time_t);
        
        const std::array<std::string, 4> positions{"front", "back", "left", "right"};
        std::vector<fs::path> file_paths;
        
        // Save images in parallel
        std::vector<int> indices{0, 1, 2, 3};
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [&](int i) {
                if (!images[i].empty()) {
                    auto filename = std::format("/home/phoenix/{}_{}_{}_{}.jpg", 
                                              g_tractor_id.load(), positions[i], timestamp, i);
                    cv::imwrite(filename, images[i]);
                    file_paths.emplace_back(filename);
                }
            });
        
        // Upload files (simplified version)
        upload_files_to_server(file_paths);
        
        // Cleanup
        for (const auto& path : file_paths) {
            fs::remove(path);
        }
    }
    
    void upload_files_to_server(const std::vector<fs::path>& file_paths) {
        // Modern string building with string_view
        std::string_view upload_command = "sshpass -p Ydl@1234567 scp {} administrator@61.48.133.19:/E:/dataset/";
        
        for (const auto& path : file_paths) {
            // Use std::format when available (C++20) or fallback
            std::string cmd = "sshpass -p Ydl@1234567 scp " + path.string() + 
                             " administrator@61.48.133.19:/E:/dataset/";
            
            auto result = std::system(cmd.c_str());
            if (result != 0) {
                g_logger.log_info("Failed to upload file: %s", path.c_str());
            }
        }
    }
};

// Log upload handler with promise/future
class LogUploadHandler {
public:
    std::future<bool> upload_log_async(std::string_view directory, std::string_view filename) {
        auto promise = std::make_shared<std::promise<bool>>();
        auto future = promise->get_future();
        
        // Launch async task
        std::thread([promise, dir = std::string(directory), fname = std::string(filename)]() {
            try {
                bool result = perform_log_upload(dir, fname);
                promise->set_value(result);
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }).detach();
        
        return future;
    }
    
private:
    static bool perform_log_upload(const std::string& directory, const std::string& filename) {
        g_logger.log_info("Starting log upload - dir: %s, filename: %s", 
                         directory.c_str(), filename.c_str());
        
        // Create upload directory
        fs::create_directories("/home/phoenix/zip");
        
        // Build zip file path
        const auto zip_path = fs::path("/home/phoenix/zip") / filename;
        
        // Create zip archive
        const std::string zip_cmd = std::format("zip -r {} /home/phoenix/log", zip_path.string());
        if (std::system(zip_cmd.c_str()) != 0) {
            g_logger.log_info("Failed to create zip archive");
            return false;
        }
        
        // Upload file (simplified implementation)
        const std::string upload_cmd = std::format("curl -X POST -F 'type=log' -F 'dir={}' -F 'file=@{}' http://server/upload", 
                                                  directory, zip_path.string());
        
        bool success = std::system(upload_cmd.c_str()) == 0;
        
        // Cleanup
        fs::remove(zip_path);
        
        g_logger.log_info("Log upload %s", success ? "completed" : "failed");
        return success;
    }
};

// Message handlers with modern C++ features
class MessageHandler {
private:
    LogUploadHandler m_log_handler;
    
public:
    void handle_tx2_message(std::string_view message) {
        g_logger.log_info("【TX2 Message】Received: %s", message.data());
        
        Json::Value json;
        Json::Reader reader;
        if (!reader.parse(message.begin(), message.end(), json)) {
            g_logger.log_info("JSON parse error");
            return;
        }
        
        const auto request = json["req"].asString();
        
        // Use string_view for efficient string matching
        if (request == "hbTX2") {
            handle_heartbeat(json);
        } else if (request == "reLogin") {
            send_registration();
        } else if (request == "update_tx2") {
            handle_update(json);
        } else if (request == "snapshot") {
            handle_snapshot(json);
        } else if (request == "uploadLog") {
            handle_log_upload(json);
        } else {
            g_logger.log_info("Unknown request: %s", request.c_str());
        }
        
        g_hb_seq++;
    }
    
private:
    void handle_heartbeat(const Json::Value& json) {
        if (g_hb_seq == 0) {
            send_registration();
        } else {
            send_heartbeat_response();
        }
        
        // Process camera streaming requests
        for (const auto& camera : json["camera"]) {
            if (auto position = camera["position"].asInt(); position >= 1 && position <= 4) {
                g_logger.log_info("Got stream position: %d", position);
                g_stream_time[position - 1] = std::time(nullptr);
                
                if (camera.isMember("url")) {
                    const auto url = camera["url"].asString();
                    setup_camera_stream(position - 1, url);
                }
            }
        }
    }
    
    void setup_camera_stream(int camera_id, std::string_view url) {
        g_logger.log_info("Setting up camera stream %d with URL: %s", camera_id, url.data());
        // Camera streaming setup implementation
    }
    
    void send_registration() {
        Json::Value msg;
        msg["req"] = "TX2Login";
        msg["seq"] = g_hb_seq.load();
        msg["ver"] = "1.7.1";
        
        for (int i = 0; i < 4; ++i) {
            msg["camera"][i]["position"] = std::to_string(i + 1);
            msg["camera"][i]["type"] = "camera";
        }
        
        Json::FastWriter writer;
        auto message = writer.write(msg);
        g_logger.log_info("【TX2 Registration】Sending: %s", message.c_str());
    }
    
    void send_heartbeat_response() {
        Json::Value msg;
        msg["req"] = "hbTX2Ret";
        msg["seq"] = g_hb_seq.load();
        
        Json::FastWriter writer;
        auto message = writer.write(msg);
        g_logger.log_info("【Heartbeat Response】Sending: %s", message.c_str());
    }
    
    void handle_snapshot(const Json::Value& json) {
        auto position = json["position"].asInt();
        if (position < 1 || position > 5) {
            g_logger.log_info("Invalid snapshot position: %d", position);
            return;
        }
        
        Json::Value response;
        response["parm"] = json["parm"];
        response["seq"] = json["seq"];
        response["req"] = "snapshotRet";
        response["position"] = json["position"];
        
        // Handle snapshot capture
        if (position <= 4) {
            std::lock_guard<std::mutex> lock(g_img_mutex);
            if (!g_img_safe[position - 1].empty()) {
                const auto snapshot_path = std::format("/tmp/snapshot_{}.jpg", position);
                cv::imwrite(snapshot_path, g_img_safe[position - 1]);
                
                if (auto encoded_data = g_base64_encoder->encode_file(snapshot_path)) {
                    response["picData"] = *encoded_data;
                    fs::remove(snapshot_path);
                }
            }
        } else if (position == 5) {
            // Radar snapshot
            std::lock_guard<std::mutex> lock(g_radar_mutex);
            if (!g_out_radar_safe.empty()) {
                const auto snapshot_path = "/tmp/snapshot_radar.jpg";
                cv::imwrite(snapshot_path, g_out_radar_safe);
                
                if (auto encoded_data = g_base64_encoder->encode_file(snapshot_path)) {
                    response["picData"] = *encoded_data;
                    fs::remove(snapshot_path);
                }
            }
        }
        
        Json::FastWriter writer;
        auto message = writer.write(response);
        g_logger.log_info("【Snapshot Response】Sending snapshot data, size: %zu bytes", message.size());
    }
    
    void handle_update(const Json::Value& json) {
        g_logger.log_info("Handling system update...");
        
        Json::Value response;
        response["req"] = "update_tx2Ret";
        response["seq"] = 222;
        response["port"] = 6669;
        
        const auto file_length = json["fileSize"].asInt();
        g_logger.log_info("Update file length: %d", file_length);
        
        // Launch update in separate thread
        std::thread([this, file_length]() {
            perform_system_update(file_length);
        }).detach();
    }
    
    void perform_system_update(int expected_file_length) {
        try {
            // Create socket for receiving update file
            SocketWrapper client_socket(AF_INET, SOCK_STREAM);
            
            sockaddr_in server_addr{};
            server_addr.sin_family = AF_INET;
            inet_aton("192.168.1.100", &server_addr.sin_addr); // Example IP
            server_addr.sin_port = htons(8080);
            
            client_socket.connect_to(reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr));
            
            // Receive and save update file
            const auto update_file = "/home/phoenix/transfer_file.zip";
            std::ofstream file(update_file, std::ios::binary);
            
            std::array<char, 4096> buffer;
            int total_received = 0;
            
            while (true) {
                auto bytes_read = client_socket.read_data(buffer.data(), buffer.size());
                if (bytes_read <= 0) break;
                
                file.write(buffer.data(), bytes_read);
                total_received += bytes_read;
            }
            
            file.close();
            
            if (total_received == expected_file_length) {
                g_logger.log_info("Update file received successfully, applying update...");
                std::system("sudo unzip -o /home/phoenix/transfer_file.zip -d /home/phoenix/tmp/");
                std::system("sh /usr/local/bin/tx2/app_update.sh");
            } else {
                g_logger.log_info("Update file length mismatch: expected %d, received %d", 
                                expected_file_length, total_received);
            }
            
        } catch (const std::exception& e) {
            g_logger.log_info("Update failed: %s", e.what());
        }
    }
    
    void handle_log_upload(const Json::Value& json) {
        const auto directory = json["dir"].asString();
        const auto filename = json["fileName"].asString();
        
        g_logger.log_info("Received log upload request - dir: %s, file: %s", 
                         directory.c_str(), filename.c_str());
        
        // Use async upload
        auto upload_future = m_log_handler.upload_log_async(directory, filename);
        
        // Don't block - the upload will complete asynchronously
    }
};

// Main application class that orchestrates all components
class PhoenixApplication {
private:
    // Modern component management using unique_ptr
    std::unique_ptr<CameraAnalyzer> m_camera_analyzer;
    std::unique_ptr<RadarReceiver> m_radar_receiver;
    std::unique_ptr<RadarProcessor> m_radar_processor;
    std::unique_ptr<ImageCaptureManager> m_image_capture;
    std::unique_ptr<FileUploadManager> m_file_upload;
    std::unique_ptr<MessageHandler> m_message_handler;
    
    std::atomic<bool> m_running{true};
    
public:
    PhoenixApplication() {
        g_logger.log_info("=== Phoenix Radar-Camera Fusion System Starting ===");
        g_logger.log_info("【Notice】eCAL communication disabled, using console output");
        
        initialize_system();
        create_components();
        start_components();
    }
    
    ~PhoenixApplication() {
        shutdown();
    }
    
    void run() {
        g_logger.log_info("【Thread Startup】All worker threads started, entering main loop...");
        
        // Main processing loop
        while (m_running && !g_shutdown_requested) {
            process_main_loop();
            std::this_thread::sleep_for(20ms);
        }
        
        g_logger.log_info("【Application】Main loop terminated");
    }
    
    void shutdown() {
        g_logger.log_info("【Application】Shutting down...");
        
        g_shutdown_requested = true;
        m_running = false;
        
        // Stop all components in reverse order
        m_file_upload.reset();
        m_image_capture.reset();
        m_radar_processor.reset();
        m_radar_receiver.reset();
        m_camera_analyzer.reset();
        
        g_logger.log_info("【Application】Shutdown complete");
    }
    
private:
    void initialize_system() {
        // Initialize logging system
        InitLog();
        // IDC_SetLoggerIP(inet_addr("182.92.85.108"), 4114);
        
        // Clean up old logs and create new log directory
        std::system("sudo rm -rf /home/phoenix/log/");
        fs::create_directories("/home/phoenix/log/");
        
        g_logger.log_info("System initialization complete");
    }
    
    void create_components() {
        // Camera paths - these would come from configuration
        std::array<std::string, 4> cam_paths = {
            "/dev/video0", "/dev/video1", "/dev/video2", "/dev/video3"
        };
        
        // Create all components using make_unique (C++14/17)
        m_camera_analyzer = std::make_unique<CameraAnalyzer>();
        m_radar_receiver = std::make_unique<RadarReceiver>();
        m_radar_processor = std::make_unique<RadarProcessor>();
        m_image_capture = std::make_unique<ImageCaptureManager>(std::move(cam_paths));
        m_file_upload = std::make_unique<FileUploadManager>();
        m_message_handler = std::make_unique<MessageHandler>();
        
        g_logger.log_info("All components created successfully");
    }
    
    void start_components() {
        // Components start automatically in their constructors
        g_logger.log_info("All components started");
    }
    
    void process_main_loop() {
        // Check if all images are available
        bool all_images_ready = false;
        {
            std::lock_guard<std::mutex> lock(g_img_mutex);
            all_images_ready = std::all_of(g_img_safe.begin(), g_img_safe.end(),
                                         [](const cv::Mat& img) { return !img.empty(); });
        }
        
        if (!all_images_ready) {
            return;
        }
        
        // Process camera detection
        std::vector<cv::Mat> img_batch;
        {
            std::lock_guard<std::mutex> lock(g_img_mutex);
            img_batch.reserve(4);
            for (const auto& img : g_img_safe) {
                img_batch.emplace_back(img.clone());
            }
        }
        
        // Run camera detection
        auto cam_det_res = g_cam_stream->run_frame(img_batch);
        publish_camera_results(cam_det_res);
        
        // Extract detection flags
        auto cam_human_flag = static_cast<int>(cam_det_res[0] - '0');
        auto cam_others_flag = static_cast<int>(cam_det_res[1] - '0');
        
        // These flags could be used for further processing
        // (Implementation would depend on specific requirements)
    }
    
    void publish_camera_results(char* cam_det_res) {
        if (!cam_det_res || strlen(cam_det_res) < 8) return;
        
        const std::array<std::string_view, 4> positions{"Front", "Back", "Left", "Right"};
        const std::array<std::string_view, 4> channels{"front_cam", "back_cam", "left_cam", "right_cam"};
        
        for (int i = 0; i < 4; ++i) {
            std::string_view result(cam_det_res + i * 2, 2);
            g_logger.log_info("【Camera Detection】%s camera detection result: %s -> %s", 
                            positions[i].data(), result.data(), channels[i].data());
        }
    }
};

// Entry point with modern exception handling
int main(int argc, char** argv) {
    try {
        // Create and run the application
        PhoenixApplication app;
        
        // Set up signal handlers for graceful shutdown
        std::signal(SIGINT, [](int) { g_shutdown_requested = true; });
        std::signal(SIGTERM, [](int) { g_shutdown_requested = true; });
        
        app.run();
        
    } catch (const std::exception& e) {
        g_logger.log_info("Application error: %s", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        g_logger.log_info("Unknown application error occurred");
        return EXIT_FAILURE;
    }
    
    g_logger.log_info("【Program End】System exit");
    return EXIT_SUCCESS;
}
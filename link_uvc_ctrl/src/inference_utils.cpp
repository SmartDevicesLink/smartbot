//
// Created by qqice on 24-6-2.
//
#include "inference_utils.h"
#include <optional> 
#include "rdk_utils.h"

#ifdef USE_RDK
#include "rdk_utils.h"
#else
#include "image_process.h"
#include "rknn_pool.h"
#endif

// C/C++ Standard Libraries for Serial Communication
#include <fcntl.h>   // 包含文件控制定义
#include <termios.h> // 包含 POSIX 终端控制定义
#include <unistd.h>  // 包含 read(), write(), close()
#include <cerrno>    // 包含 errno 变量

// --- 1. 定义一个简单可靠的通信协议 ---
// 作用：确保接收端（单片机）能够准确地识别数据包的开始和结束，并校验数据完整性。
#pragma pack(1) // 按1字节对齐，确保结构体大小正确
struct ControlPacket {
    const uint8_t header = 0xFF;  // 帧头
    int16_t dx = 0;               // x方向偏移量 (-32768 to 32767)
    int16_t dy = 0;               // y方向偏移量
    uint8_t checksum = 0;         // 校验和
    const uint8_t footer = 0xFE;  // 帧尾

    // 计算校验和的方法
    void calculate_checksum() {
        checksum = 0;
        checksum ^= (dx >> 8) & 0xFF; // dx 高8位
        checksum ^= dx & 0xFF;        // dx 低8位
        checksum ^= (dy >> 8) & 0xFF; // dy 高8位
        checksum ^= dy & 0xFF;        // dy 低8位
    }
};
#pragma pack() // 恢复默认对齐

// --- 2. 创建一个串口工具类 ---
class SerialPort {
public:
    SerialPort() = default;
    ~SerialPort() {
        if (is_open()) {
            close(fd_);
        }
    }

    bool open_port(const std::string& port_name, int baud_rate) {
        fd_ = open(port_name.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd_ < 0) {
            spdlog::error("Error {} opening {}: {}", errno, port_name, strerror(errno));
            return false;
        }

        struct termios options;
        tcgetattr(fd_, &options);

        // 设置波特率
        cfsetispeed(&options, baud_rate);
        cfsetospeed(&options, baud_rate);

        // 设置模式为 8N1 (8数据位, 无校验, 1停止位)
        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;
        options.c_cflag &= ~PARENB;
        options.c_cflag &= ~CSTOPB;

        // 设置为原始模式 (raw mode)
        cfmakeraw(&options);

        // 刷新端口并应用设置
        tcflush(fd_, TCIFLUSH);
        if (tcsetattr(fd_, TCSANOW, &options) != 0) {
            spdlog::error("Error {} from tcsetattr: {}", errno, strerror(errno));
            return false;
        }
        
        spdlog::info("Serial port {} opened successfully at {} baud.", port_name, baud_rate);
        return true;
    }

    bool write_data(const void* data, size_t length) {
        if (!is_open()) return false;
        ssize_t bytes_written = write(fd_, data, length);
        if (bytes_written < 0) {
            spdlog::error("Error {} writing to serial port: {}", errno, strerror(errno));
            return false;
        }
        return bytes_written == length;
    }

    bool is_open() const {
        return fd_ >= 0;
    }

private:
    int fd_ = -1; // 文件描述符
};

// RDK模型路径，使用.bin格式而不是.rknn
#ifdef USE_RDK
std::string model_path = "../model/yolo11x_detect_bayese_640x640_nv12_modified.bin";
#else
std::string model_path = "../model/yolov8n.rknn";
#endif
std::string label_path = "../model/coco80labels.txt";
int thread_num = 1;

#ifdef USE_RDK
// RDK推理实例
std::unique_ptr<RDKInference> rdk_inference;
#endif

/**
 * @brief 从检测结果中找到置信度最高的人，并计算其中心与图像中心的偏移量。
 *
 * @param results 来自 RDKInference::runInference 的检测结果列表。
 * @param imageWidth 原始图像的宽度。
 * @param imageHeight 原始图像的高度。
 * @return std::optional<cv::Point2f>
 * - 如果找到人，则返回一个包含(dx, dy)偏移量的点。
 * - 如果未找到人，则返回 std::nullopt。
 */

std::optional<std::pair<float, float>> calculatePersonCenterOffset(
    const std::vector<DetectionResult>& results,
    int imageWidth,
    int imageHeight) 
{
    const DetectionResult* bestPersonResult = nullptr;
    // float maxConfidence = 0.0f;
    float maxArea = 0.0f;

    // 1. 遍历所有检测结果，筛选出置信度最高的 "person"
    for (const auto& result : results) {
        if (result.class_name == "person") {
            float width = result.x2 - result.x1;
            float height = result.y2 - result.y1;
            float currentArea = width * height;
            if (currentArea > maxArea) {
                maxArea = currentArea;         // 更新最大面积
                bestPersonResult = &result;  // 记录下这个面积最大的人
            }
        }
    }

    // 2. 如果成功找到了人
    if (bestPersonResult) {
        // 计算人物框的中心点
        float personCenterX = (bestPersonResult->x1 + bestPersonResult->x2) / 2.0f;
        float personCenterY = (bestPersonResult->y1 + bestPersonResult->y2) / 2.0f;
        
        // 计算图像的中心点
        float imageCenterX = imageWidth / 2.0f;
        float imageCenterY = imageHeight / 2.0f;
        
        // 计算差值（偏移量）
        float offsetX = personCenterX - imageCenterX;
        float offsetY = personCenterY - imageCenterY;
        
        // ⭐ 主要改动在这里：返回一个 std::pair 而不是 cv::Point2f
        return std::make_pair(offsetX, offsetY);
    }

    // 3. 如果没有找到人，返回空值
    return std::make_pair(0.0f, 0.0f); // 返回图像中心作为默认值
}

void inference_thread() {
#ifdef USE_RDK
    try {
        // 初始化RDK推理引擎
        rdk_inference = std::make_unique<RDKInference>(model_path);
        spdlog::info("RDK Inference initialized successfully");
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize RDK Inference: {}", e.what());
        return;
    }
    
    spdlog::info("RDK inference thread started");

    SerialPort serial_port;
    // !! 重要: 请将 "/dev/ttyUSB0" 替换成你实际的串口设备名 !!
    // 你可以通过 `ls /dev/tty*` 命令查找
    if (!serial_port.open_port("/dev/ttyUSB0", B115200)) {
        spdlog::error("Could not open serial port. Tracking commands will not be sent.");
        // 这里可以选择直接返回，或者让程序继续运行但不发送串口数据
    }
    
    while (loopFlag) {

        if (!g_auto_tracking_enabled.load()) {
            // 如果已暂停，则休眠并跳过所有工作
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue; 
        }

        auto loop_start_time = std::chrono::steady_clock::now();
        // if (need_inference.load()) { // 收到"推理"请求
        // spdlog::info("Inference_thread received inference request");
        std::unique_ptr<cv::Mat> image = std::make_unique<cv::Mat>();
        
        if (frame_available.load()) { // 检查帧可用标志
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                *image = current_frame.clone(); // 获取当前帧副本
            }
            frame_available.store(false); // 重置帧可用标志
            
            if (!image->empty()) {
                spdlog::info("Start RDK inference");
                
                try {
                    // 使用RDK推理引擎进行推理
                    std::vector<DetectionResult> detection_results;
                    int ret = rdk_inference->runInference(*image, detection_results);
                    
                    if (ret == 0) {
                        spdlog::info("RDK Inference finished, detected {} objects", detection_results.size());

                        auto offsetResult = calculatePersonCenterOffset(
                            detection_results, 
                            image->cols, 
                            image->rows
                        );
                        
                        if (offsetResult.has_value()) {
                            std::pair<float, float> offset = offsetResult.value();
                            const float k = 0.1f; 
                            int dx = static_cast<int>(std::round(offset.first * k));
                            int dy = static_cast<int>(std::round(offset.second * k));
                            std::string dx_str = std::to_string(dx);
                            std::string dy_str = std::to_string(dy);
                            std::string command = dx_str + " " + dy_str + "\n";
                            // int dx = (int)offset.first;
                            // int dy = (int)offset.second;
                            if (serial_port.is_open()){
                                serial_port.write_data(command.c_str(), command.length());
                                spdlog::info("Sent UART string: {}", command.substr(0, command.length() - 1));
                            }
                            
                        }
                    } else {
                        spdlog::error("RDK inference failed with code: {}", ret);
                    }
                } catch (const std::exception& e) {
                    spdlog::error("Exception during RDK inference: {}", e.what());
                }
            } else {
                spdlog::warn("Current frame is empty, skipping inference");
            }
        } else {
            spdlog::warn("No frame available for inference");
        }

        auto loop_end_time = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end_time - loop_start_time);

        auto time_to_wait = std::chrono::milliseconds(100) - elapsed_time;
        if (time_to_wait.count() > 0) {
            std::this_thread::sleep_for(time_to_wait);
        }
            // need_inference.store(false); // 重置"推理"请求状态
        // }
        
        // // 若没有"推理"请求,则等待一段时间
        // spdlog::debug("Waiting for inference request");
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
#else
    // 原有的RKNN实现作为备选方案
    auto rknn_pool = std::make_unique<RknnPool>(
            model_path, thread_num, label_path);
    spdlog::info("RKNN pool initialized");
    while (true) {
        if (need_inference.load()) { // 收到"推理"请求
            spdlog::info("Inference_thread received inference request");
            std::unique_ptr<cv::Mat> image = std::make_unique<cv::Mat>();
            if (frame_available.load()) { // 检查帧可用标志
                {
                    std::lock_guard<std::mutex> lock(frame_mutex);
                    *image = current_frame.clone(); // 获取当前帧副本
                }
                frame_available.store(false); // 重置帧可用标志
                if (!image->empty()) {
                    spdlog::info("Start inference");
                    spdlog::debug("Preprocessing image");
                    ImageProcess image_process(image->cols, image->rows, 640);
                    std::shared_ptr<cv::Mat> image_res;
                    // 进行推理
                    rknn_pool->AddInferenceTask(std::move(image), image_process);
                    while (image_res == nullptr) {
                        image_res = rknn_pool->GetImageResultFromQueue();
                    }
                    spdlog::info("Inference finished");
                    upload_to_CF(current_name, *image_res);
                    spdlog::info("Result uploaded");
                }
            }
            need_inference.store(false); // 重置"推理"请求状态
        }
        // 若没有"推理"请求,则等待一段时间
        spdlog::debug("Waiting for inference request");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
#endif
}
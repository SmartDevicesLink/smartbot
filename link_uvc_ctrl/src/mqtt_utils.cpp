#include "mqtt_utils.h"
#include "uvc_utils.h"
#include <nlohmann/json.hpp>
#include <string> // 

// MQTT设置
const char *MQTT_HOST = "127.0.0.1";
const int MQTT_PORT = 1883;
const char *MQTT_TOPIC = "camera/control";

const char *MQTT_MODE_CONTROL_TOPIC = "camera/tracking/set_mode";

// MQTT消息回调函数
void on_message_callback(struct mosquitto *mosq, void *obj, const struct mosquitto_message *message) {
    if (!message || !message->payloadlen) {
        spdlog::error("Received an empty or invalid message.");
        return;
    }

    std::string topic = message->topic;
    std::string payload = static_cast<char*>(message->payload);
    
    spdlog::info("Received message on topic '{}': {}", topic, payload);

    // 手动控制
    if (topic == MQTT_TOPIC) {
        spdlog::info("Handling manual control command...");
        try {
            nlohmann::json jsonParsed = nlohmann::json::parse(payload);
        
            switch (jsonParsed["control"].get<int>()) {
                case CAMERA_GIMBAL_CONTROL:
                    set_camera_gimbal_control(devh, (char) jsonParsed["horizontal_direction"].get<int>(),
                                              (char) jsonParsed["horizontal_speed"].get<int>(),
                                              (char) jsonParsed["vertical_direction"].get<int>(),
                                              (char) jsonParsed["vertical_speed"].get<int>());
                    break;
                case CAMERA_GIMBAL_STOP:
                    stop_camera_gimbal_control(devh);
                    break;
                case CAMERA_GIMBAL_CENTER:
                    set_camera_gimbal_to_center(devh);
                    break;
                case CAMERA_ZOOM:
                    set_camera_zoom_absolute(devh, jsonParsed["zoom"].get<int>());
                    break;
                case CAMERA_GIMBAL_LOCATION:
                    set_camera_gimbal_location(devh, jsonParsed["horizontal_location"].get<int>(),
                                               jsonParsed["vertical_location"].get<int>(), jsonParsed["zoom"].get<int>());
                    break;
                case LEAF_DISEASE_INFERENCE:
                  {
                    spdlog::info("Received leaf disease inference request");
                    need_inference.store(true);
                    current_name = jsonParsed["name"];
                    break;
                  }
                default:
                    spdlog::error("Unknown manual control command: {}", jsonParsed["control"].get<int>());
                    break;
            }
        } catch (nlohmann::json::parse_error &e) {
            spdlog::error("JSON parse error for manual control: {}", e.what());
        }

    // 模式控制
    } else if (topic == MQTT_MODE_CONTROL_TOPIC) {
        spdlog::info("Handling mode control command...");
        try {
            nlohmann::json jsonParsed = nlohmann::json::parse(payload);
            std::string mode = jsonParsed["mode"];

            if (mode == "pause") {
                g_auto_tracking_enabled.store(false);
                spdlog::info("Auto-tracking has been PAUSED.");
            } else if (mode == "auto") {
                g_auto_tracking_enabled.store(true);
                spdlog::info("Auto-tracking has been RESUMED.");
            } else {
                spdlog::warn("Unknown mode: {}", mode);
            }
        } catch (nlohmann::json::parse_error &e) {
            spdlog::error("JSON parse error for mode control: {}", e.what());
        }
    }
}

// MQTT循环处理函数
void mqtt_loop(struct mosquitto *mosq) {
    mosquitto_loop_forever(mosq, -1, 1);
}
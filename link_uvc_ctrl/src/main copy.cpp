
#include <iostream>
#include <cstring>
#include <thread>
#include <csignal>
#include "mqtt_utils.h"
#include "uvc_utils.h"
#include "spdlog/spdlog.h"
#include "inference_utils.h"

#ifdef USE_REALSENSE
#include "realsense_utils.h"
#endif


// 标志变量，用来控制循环
volatile sig_atomic_t loopFlag = 1;

// 信号处理函数
void signalHandler(int signal) {
    if (signal == SIGINT) {
        // 当接收到SIGINT信号时，修改循环控制变量
        loopFlag = 0;
    }
}


int main(int argc, char **argv) {
    uvc_context_t *ctx;
    uvc_device_t *dev;
    uvc_stream_ctrl_t ctrl;
    uvc_error_t res;
    //setenv("GST_DEBUG", "3", 1);

    //std::cout << cv::getBuildInformation() << std::endl;

    // 注册信号处理函数
    signal(SIGINT, signalHandler);

    // 初始化mosquitto库
    mosquitto_lib_init();

    // 创建新的mosquitto客户端实例
    struct mosquitto *mosq = mosquitto_new(nullptr, true, nullptr);
    if (!mosq) {
        spdlog::error("Failed to create mosquitto instance.");
        return -1;
    }

    // 设置消息回调
    mosquitto_message_callback_set(mosq, on_message_callback);

    // 连接到MQTT代理服务器
    if (mosquitto_connect(mosq, MQTT_HOST, MQTT_PORT, 60)) {
        spdlog::error("Could not connect to MQTT Broker.");
        return -1;
    }

    // 订阅主题
    mosquitto_subscribe(mosq, nullptr, MQTT_TOPIC, 0);

    // 创建并启动处理MQTT消息的线程
    std::thread mqtt_thread(mqtt_loop, mosq);

#ifdef USE_REALSENSE
    // 创建并启动Realsense线程
    std::thread realsense_thread(realsense_loop);
#endif

    // 创建并启动推理线程
    std::thread inf_thread(inference_thread);

    /* Initialize a UVC service context. Libuvc will set up its own libusb
     * context. Replace NULL with a libusb_context pointer to run libuvc
     * from an existing libusb context. */
    res = uvc_init(&ctx, nullptr);

    if (res < 0) {
        spdlog::error("uvc_init failed");
        return res;
    }

    spdlog::info("UVC initialized");

    /* Locates the first attached UVC device, stores in dev */
    res = uvc_find_device(
            ctx, &dev,
            0x2e1a, 0x4c01, nullptr); /* filter devices: vendor_id, product_id, "serial_num" */

    if (res < 0) {
        spdlog::error("uvc_find_device failed");
    } else {
        spdlog::info("Device found");

        /* Try to open the device: requires exclusive access */
        res = uvc_open(dev, &devh);

        if (res < 0) {
            spdlog::error("uvc_open failed");
        } else {
            spdlog::info("Device opened");

            /* Print out a message containing all the information that libuvc
             * knows about the device */
            //uvc_print_diag(devh, stderr);

            const uvc_format_desc_t *format_desc = uvc_get_format_descs(devh);
            const uvc_frame_desc_t *frame_desc = format_desc->frame_descs;
            enum uvc_frame_format frame_format;

            switch (format_desc->bDescriptorSubtype) {
                case UVC_VS_FORMAT_MJPEG:
                    frame_format = UVC_COLOR_FORMAT_MJPEG;
                    break;
                case UVC_VS_FORMAT_FRAME_BASED:
                    frame_format = UVC_FRAME_FORMAT_H264;
                    break;
                default:
                    frame_format = UVC_FRAME_FORMAT_YUYV;
                    break;
            }

            if (frame_desc) {
                width = frame_desc->wWidth;
                height = frame_desc->wHeight;
                fps = 10000000 / frame_desc->dwDefaultFrameInterval;
            }

            spdlog::debug("First format: ({}) {}x{} {}fps", format_desc->fourccFormat, width, height, fps);
            /* Try to negotiate first stream profile */
            res = uvc_get_stream_ctrl_format_size(
                    devh, &ctrl, /* result stored in ctrl */
                    frame_format,
                    width, height, fps /* width, height, fps */
            );

            /* Print out the result */
            //uvc_print_stream_ctrl(&ctrl, stderr);

            if (res < 0) {
                spdlog::error("get_mode failed");
            } else {
                /* Start the video stream. The library will call user function cb:
                 *   cb(frame, (void *) 12345)
                 */
                res = uvc_start_streaming(devh, &ctrl, cb, (void *) 12345, 0);

                if (res < 0) {
                    spdlog::error("start_streaming failed");
                } else {
                    spdlog::info("Streaming...");

                    // sleep(600); /* stream for 10 minutes */

                    while (loopFlag) {
                        sleep(1);
                    }
                    // 等待MQTT线程结束（在这个示例中，线程将无限循环，因此下面的join调用实际上会阻塞）
                    // mqtt_thread.join();

                    /* End the stream. Blocks until last callback is serviced */
                    uvc_stop_streaming(devh);
                    mosquitto_destroy(mosq);
                    spdlog::info("Done streaming.");
                }
            }

            /* Release our handle on the device */
            uvc_close(devh);
            spdlog::info("Device closed");
        }

        /* Release the device descriptor */
        uvc_unref_device(dev);
    }

    /* Close the UVC context. This closes and cleans up any existing device handles,
     * and it closes the libusb context if one was not provided. */
    uvc_exit(ctx);
    spdlog::info("UVC exited");
    mosquitto_lib_cleanup();
    return 0;
}

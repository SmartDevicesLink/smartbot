#include "rdk_utils.h"    // 引入你的头文件以获取 DetectionResult 的定义
#include <optional>         // 用于处理可能不存在的返回值

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
std::optional<cv::Point2f> calculatePersonCenterOffset(
    const std::vector<DetectionResult>& results,
    int imageWidth,
    int imageHeight) 
{
    const DetectionResult* bestPersonResult = nullptr;
    float maxConfidence = 0.0f;

    // 1. 遍历所有检测结果，筛选出置信度最高的 "person"
    for (const auto& result : results) {
        // 使用 "person" 字符串进行判断
        if (result.class_name == "person") {
            if (result.confidence > maxConfidence) {
                maxConfidence = result.confidence;
                bestPersonResult = &result; // 记录下这个最佳目标的地址
            }
        }
    }

    // 2. 如果成功找到了人 (bestPersonResult 不再是空指针)
    if (bestPersonResult) {
        // 3. 计算目标人物框的中心点
        float personCenterX = (bestPersonResult->x1 + bestPersonResult->x2) / 2.0f;
        float personCenterY = (bestPersonResult->y1 + bestPersonResult->y2) / 2.0f;
        
        // 4. 计算图像的中心点
        float imageCenterX = imageWidth / 2.0f;
        float imageCenterY = imageHeight / 2.0f;
        
        // 5. 计算两个中心点的差值 (偏移量)
        cv::Point2f offset(personCenterX - imageCenterX, personCenterY - imageCenterY);
        
        // 6. 返回这个差值
        return offset;
    }

    // 如果循环结束都没有找到 "person" 目标，则返回一个表示“空”的可选值
    return std::nullopt;
}
import subprocess
import paho.mqtt.client as mqtt
import json
import time
import os
import signal

# ======================= 1. 配置信息 =======================
# 请将这里的IP地址修改为你MQTT服务器的实际IP地址
MQTT_BROKER_HOST = "192.168.137.1" 
MQTT_BROKER_PORT = 1883

# --- 路径配置 (!! 非常重要，请根据你的实际路径修改 !!) ---
# C++ "自动进程" 可执行文件的绝对路径
PATH_TO_AUTO_PROCESS_EXECUTABLE = "/root/insta360_link_uvc_ctrl/build/V4L2Capture"
# Python "手动进程" 脚本的绝对路径
PATH_TO_MANUAL_PROCESS_SCRIPT = "/root/hand/gimbal_device_simulator.py --host 192.168.137.1 --port 1883"

# --- MQTT 主题配置 ---
# 总指挥用来接收模式切换指令的主题
MANAGER_TOPIC = "camera/manager/set_mode"
# 用于控制C++自动进程“暂停/恢复”的主题
AUTO_PROCESS_CONTROL_TOPIC = "camera/tracking/set_mode"
# ==========================================================

# --- 全局状态变量 ---
current_mode = "idle"  # 当前模式: idle, auto, manual
auto_process = None    # 存放自动进程的Popen对象
manual_process = None  # 存放手动进程的Popen对象

# 全局MQTT客户端
mqtt_client = mqtt.Client()

def start_auto_process():
    """启动C++自动进程，并告诉它进入'auto'模式"""
    global auto_process, current_mode
    if auto_process is None or auto_process.poll() is not None:
        print("[管理器]: 正在启动'自动进程'...")
        # 获取可执行文件所在的目录
        cwd = os.path.dirname(PATH_TO_AUTO_PROCESS_EXECUTABLE)
        auto_process = subprocess.Popen([PATH_TO_AUTO_PROCESS_EXECUTABLE], cwd=cwd)
        print(f"[管理器]: '自动进程'已启动 (PID: {auto_process.pid})。")
        # 等待一下，确保它有足够时间订阅MQTT
        time.sleep(2)
    
    # 发送指令让它开始追踪
    mqtt_client.publish(AUTO_PROCESS_CONTROL_TOPIC, '{"mode":"auto"}', qos=1)
    print(f"[管理器]: 已向'自动进程'发送'auto'指令。")
    current_mode = "auto"

def stop_auto_process():
    """暂停C++自动进程（不杀死它，而是让它待机）"""
    global current_mode
    # 发送指令让它停止追踪
    mqtt_client.publish(AUTO_PROCESS_CONTROL_TOPIC, '{"mode":"pause"}', qos=1)
    print(f"[管理器]: 已向'自动进程'发送'pause'指令。")
    if current_mode == "auto":
        current_mode = "idle"

def start_manual_process():
    """启动Python手动进程"""
    global manual_process, current_mode
    if manual_process is None or manual_process.poll() is not None:
        print("[管理器]: 正在启动'手动进程'...")
        manual_process = subprocess.Popen(["python3", PATH_TO_MANUAL_PROCESS_SCRIPT])
        print(f"[管理器]: '手动进程'已启动 (PID: {manual_process.pid})。")
    current_mode = "manual"

def stop_manual_process():
    """终止Python手动进程"""
    global manual_process, current_mode
    if manual_process and manual_process.poll() is None:
        print(f"[管理器]: 正在终止'手动进程' (PID: {manual_process.pid})...")
        manual_process.terminate() # 发送终止信号
        manual_process.wait() # 等待进程结束
        print("[管理器]: '手动进程'已终止。")
    if current_mode == "manual":
        current_mode = "idle"

# MQTT回调函数
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[管理器]: 成功连接到MQTT服务器。")
        client.subscribe(MANAGER_TOPIC)
        print(f"[管理器]: 已订阅管理主题: {MANAGER_TOPIC}")
    else:
        print(f"[管理器]: 连接失败，返回码: {rc}")

def on_message(client, userdata, msg):
    """处理模式切换指令"""
    global current_mode
    payload_str = msg.payload.decode('utf-8')
    print(f"\n[管理器]: 收到模式切换指令 -> {payload_str}")

    try:
        data = json.loads(payload_str)
        target_mode = data.get("mode")

        if target_mode == "manual":
            if current_mode != "manual":
                print("[管理器]: 正在切换到手动模式...")
                stop_auto_process()  # 先暂停自动进程
                start_manual_process() # 再启动手动进程
                print("[管理器]: 已切换到手动模式。")
            else:
                print("[管理器]: 已处于手动模式，无需切换。")

        elif target_mode == "auto":
            if current_mode != "auto":
                print("[管理器]: 正在切换到自动模式...")
                stop_manual_process() # 先停止手动进程
                start_auto_process()  # 再恢复自动进程
                print("[管理器]: 已切换到自动模式。")
            else:
                print("[管理器]: 已处于自动模式，无需切换。")
        
        else:
            print(f"[管理器]: 未知的模式指令: {target_mode}")

    except Exception as e:
        print(f"[管理器]: 处理指令时出错: {e}")


# --- 主程序 ---
def main():
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    try:
        mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    except Exception as e:
        print(f"[管理器]: 无法连接到MQTT服务器: {e}")
        return

    # 默认启动时，先启动自动进程
    start_auto_process()

    try:
        mqtt_client.loop_forever()
    except KeyboardInterrupt:
        print("\n[管理器]: 程序被用户中断。")
    finally:
        print("[管理器]: 正在清理所有子进程...")
        # 确保所有子进程都被关闭
        if auto_process and auto_process.poll() is None:
            auto_process.terminate()
            auto_process.wait()
        if manual_process and manual_process.poll() is None:
            manual_process.terminate()
            manual_process.wait()
        mqtt_client.disconnect()
        print("[管理器]: 清理完毕，程序退出。")

if __name__ == "__main__":
    main()
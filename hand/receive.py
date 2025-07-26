import paho.mqtt.client as mqtt
import json
import time
import serial  # 引入pyserial库

# ======================= 1. 配置信息 =======================
# 请将这里的IP地址修改为你MQTT服务器的实际IP地址
MQTT_BROKER_HOST = "192.168.137.1" 
MQTT_BROKER_PORT = 1883
MQTT_MANUAL_CONTROL_TOPIC = "device/gimbal/control"

# --- 新增：串口配置 ---
# !! 重要: 请将 "/dev/ttyUSB0" 替换成你实际的串口设备名 !!
# 在Linux上通常是 /dev/ttyUSB0, /dev/ttyS0, /dev/ttyAMA0 等
# 在Windows上是 "COM3", "COM4" 等
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
# ==========================================================

# 全局变量，用于保存串口对象
serial_port = None

# 当成功连接到MQTT服务器时会调用的函数 (此函数不变)
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"成功连接到MQTT服务器 (Code: {rc})")
        client.subscribe(MQTT_MANUAL_CONTROL_TOPIC)
        print(f"已订阅主题: {MQTT_MANUAL_CONTROL_TOPIC}")
    else:
        print(f"连接失败，返回码: {rc}")

# 当收到订阅主题的消息时会调用的函数
def on_message(client, userdata, msg):
    """处理接收到的消息，并将其转换为串口指令发送出去。"""
    payload_str = msg.payload.decode('utf-8')
    print(f"\n收到新消息 -> 主题: {msg.topic} | 内容: {payload_str}")

    serial_command = None  # 用于存放最终要发送的串口指令

    try:
        data = json.loads(payload_str)
        command = data.get("command")

        if command == "move":
            direction = data.get("direction", "stop").lower()
            speed = data.get("speed", 0)
            
            # 将方向字符串转换为单个字符，方便单片机处理
            dir_char_map = {"up": "U", "down": "D", "left": "L", "right": "R", "stop": "S"}
            dir_char = dir_char_map.get(direction, "S") # 默认为停止
            
            serial_command = f"M,{dir_char},{speed}\n"

        elif command == "zoom":
            level = data.get("level", 100)
            serial_command = f"Z,{level}\n"

        # --- 在这里可以添加更多指令的转换 ---

        else:
            print(f"  [警告]: 收到未知的指令 '{command}'")

    except Exception as e:
        print(f"  [错误]: 处理消息时发生异常: {e}")
        return

    # --- 将转换后的指令通过串口发送 ---
    if serial_command and serial_port and serial_port.is_open:
        try:
            # 字符串必须先编码成字节(bytes)才能发送
            encoded_command = serial_command.encode('utf-8')
            serial_port.write(encoded_command)
            # .strip() 用于移除末尾的换行符，让打印更整洁
            print(f"  [串口发送]: 成功发送 -> {serial_command.strip()}")
        except Exception as e:
            print(f"  [串口错误]: 发送失败: {e}")
    elif serial_command:
        print("  [错误]: 串口未打开，无法发送指令。")


# --- 主程序 ---
def main():
    global serial_port  # 声明我们将要修改全局的 serial_port 变量

    # --- 初始化串口 ---
    try:
        print(f"正在打开串口: {SERIAL_PORT} @ {BAUD_RATE}bps...")
        serial_port = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # 等待串口稳定
        print("串口已成功打开。")
    except serial.SerialException as e:
        print(f"!!! 无法打开串口: {e}")
        print("!!! 请检查：")
        print(f"    1. 串口名称 '{SERIAL_PORT}' 是否正确。")
        print(f"    2. 设备是否已连接到电脑。")
        print(f"    3. 是否有权限访问该串口 (在Linux上可能需要将用户添加到 'dialout' 组)。")
        return  # 如果串口打不开，程序直接退出

    # --- 初始化MQTT客户端 ---
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    print(f"正在尝试连接到MQTT服务器: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    except Exception as e:
        print(f"无法连接到MQTT服务器: {e}")
        return

    # --- 启动主循环，并确保程序退出时能安全关闭串口 ---
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    finally:
        print("正在关闭串口...")
        if serial_port and serial_port.is_open:
            serial_port.close()
        print("正在断开MQTT连接...")
        client.disconnect()
        print("程序已退出。")


if __name__ == "__main__":
    print("--- 手动控制进程已启动 (MQTT <-> Serial) ---")
    print("--- 按下 Ctrl+C 停止 ---")
    main()
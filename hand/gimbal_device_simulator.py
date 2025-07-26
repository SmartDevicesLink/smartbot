#!/usr/bin/env python3
"""
云台设备模拟器
模拟Linux云台设备通过MQTT与聊天室服务器通信

功能:
1. 连接到MQTT服务器
2. 注册云台设备用户名"云台"
3. 订阅云台控制事件 (device/gimbal/control)
4. 接收并执行云台控制指令
5. 发送状态更新
"""
import json
import logging
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import paho.mqtt.client as mqtt
import argparse
import signal
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GimbalDevice")


class GimbalDeviceSimulator:
    """云台设备模拟器"""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883, 
                 device_id: str = None, username: str = None, password: str = None):
        """
        初始化云台设备模拟器
        
        Args:
            broker_host: MQTT代理服务器地址
            broker_port: MQTT代理服务器端口
            device_id: 设备唯一标识符
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        # 如果没有提供device_id，生成唯一ID避免客户端冲突
        self.device_id = device_id if device_id else f"gimbal_{int(time.time())}"
        self.username = "云台"  # 固定用户名
        
        # MQTT身份验证参数
        self.mqtt_username = username
        self.mqtt_password = password
        
        # MQTT客户端配置
        self.client = mqtt.Client(client_id=self.device_id)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # 设备状态
        self.is_connected = False
        self.is_running = False
        self.current_position = {"x": 2036, "y": 2125}  # 初始位置
        self.position_limits = {
            "x": {"min": 1024, "max": 3048},
            "y": {"min": 1850, "max": 2400}
        }
        
        # MQTT主题配置
        self.topics = {
            'control': 'device/gimbal/control',        # 接收控制指令
            'register': 'device/gimbal/register',      # 发送注册信息
            'status': 'device/gimbal/status',          # 发送状态更新
            'chat_in': 'chatroom/messages/in'          # 发送聊天消息（可选）
        }
        
        # 统计信息
        self.stats = {
            'connect_time': None,
            'commands_received': 0,
            'commands_executed': 0,
            'position_changes': 0,
            'last_command_time': None
        }
        
        # 状态发送定时器
        self.status_timer = None
        
        logger.info(f"云台设备模拟器初始化完成: {device_id} @ {broker_host}:{broker_port}")
    
    def start(self) -> bool:
        """
        启动云台设备模拟器
        
        Returns:
            启动是否成功
        """
        try:
            if self.is_running:
                logger.warning("云台设备已在运行")
                return True
            
            logger.info(f"连接到MQTT代理: {self.broker_host}:{self.broker_port}")
            
            # 设置MQTT身份验证
            if self.mqtt_username:
                logger.info(f"使用身份验证: {self.mqtt_username}")
                self.client.username_pw_set(self.mqtt_username, self.mqtt_password)
            
            self.client.connect(self.broker_host, self.broker_port, 60)
            
            # 启动网络循环
            self.client.loop_start()
            self.is_running = True
            
            # 等待连接建立
            retry_count = 0
            while not self.is_connected and retry_count < 10:
                time.sleep(0.5)
                retry_count += 1
            
            if self.is_connected:
                logger.info("云台设备启动成功")
                
                # 注册设备
                self._register_device()
                
                # 启动状态发送定时器 (每30秒发送一次状态)
                self._start_status_timer()
                
                return True
            else:
                logger.error("MQTT连接超时")
                self.stop()
                return False
                
        except Exception as e:
            error_msg = str(e)
            if "No route to host" in error_msg or "10061" in error_msg:
                logger.error(f"云台设备启动失败: MQTT代理无法连接 ({self.broker_host}:{self.broker_port})")
                logger.error("请检查: 1) MQTT代理是否正在运行 2) 网络连接是否正常 3) IP地址和端口是否正确")
                if self.broker_host not in ["localhost", "127.0.0.1"]:
                    logger.info("建议: 在开发环境中使用 'localhost' 或 '127.0.0.1' 作为MQTT代理地址")
            elif "Connection refused" in error_msg:
                logger.error(f"云台设备启动失败: MQTT代理拒绝连接 ({self.broker_host}:{self.broker_port})")
                logger.error("请检查MQTT代理(Mosquitto)是否正在运行且监听此端口")
            else:
                logger.error(f"云台设备启动失败: {e}")
            return False
    
    def stop(self):
        """停止云台设备模拟器"""
        try:
            if not self.is_running:
                return
            
            logger.info("正在停止云台设备...")
            self.is_running = False
            
            # 停止状态定时器
            if self.status_timer:
                self.status_timer.cancel()
            
            # 发送离线状态
            if self.is_connected:
                self._send_offline_status()
                time.sleep(0.5)  # 等待消息发送
            
            # 断开连接
            self.client.loop_stop()
            self.client.disconnect()
            
            self.is_connected = False
            logger.info("云台设备已停止")
            
        except Exception as e:
            logger.error(f"云台设备停止异常: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT连接回调"""
        if rc == 0:
            self.is_connected = True
            self.stats['connect_time'] = datetime.now()
            logger.info("云台设备MQTT连接成功")
            
            # 订阅云台控制主题
            client.subscribe(self.topics['control'])
            logger.info(f"订阅云台控制主题: {self.topics['control']}")
            
        else:
            logger.error(f"MQTT连接失败，错误代码: {rc}")
            self.is_connected = False
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT断开连接回调"""
        self.is_connected = False
        if rc == 0:
            logger.info("云台设备MQTT正常断开连接")
        elif rc == 7:
            logger.warning(f"云台设备MQTT连接丢失 (错误代码: {rc}) - 可能是客户端ID冲突")
            if self.is_running:
                logger.info("正在尝试重新连接...")
        else:
            logger.warning(f"云台设备MQTT连接断开，代码: {rc}")
    
    def _on_message(self, client, userdata, message):
        """MQTT消息回调"""
        try:
            topic = message.topic
            payload = message.payload.decode('utf-8')
            
            logger.info(f"收到MQTT消息: {topic} -> {payload}")
            
            if topic == self.topics['control']:
                self._handle_control_command(payload)
            else:
                logger.warning(f"未处理的MQTT主题: {topic}")
                
        except Exception as e:
            logger.error(f"处理MQTT消息异常: {e}")
    
    def _handle_control_command(self, payload: str):
        """
        处理云台控制指令
        期望格式: "Ang_X=xxx,Ang_Y=yyy"
        
        Args:
            payload: 控制指令内容
        """
        try:
            logger.info(f"收到云台控制指令: {payload}")
            self.stats['commands_received'] += 1
            self.stats['last_command_time'] = datetime.now()
            
            # 解析指令格式
            if not self._validate_command_format(payload):
                logger.error(f"云台控制指令格式错误: {payload}")
                return
            
            # 解析角度参数
            ang_x, ang_y = self._parse_angles(payload)
            
            # 验证参数范围
            if not self._validate_angles(ang_x, ang_y):
                logger.error(f"云台控制参数超出范围: X={ang_x}, Y={ang_y}")
                return
            
            # 执行控制指令
            success = self._execute_control(ang_x, ang_y)
            
            if success:
                self.stats['commands_executed'] += 1
                self.stats['position_changes'] += 1
                
                logger.info(f"云台控制执行成功: X={ang_x}, Y={ang_y}")
                
                # 发送状态更新
                self._send_status_update()
                
                # 可选：发送聊天消息确认
                self._send_chat_confirmation(ang_x, ang_y)
            else:
                logger.error(f"云台控制执行失败: X={ang_x}, Y={ang_y}")
                
        except Exception as e:
            logger.error(f"处理云台控制指令异常: {e}")
    
    def _validate_command_format(self, payload: str) -> bool:
        """验证指令格式"""
        import re
        pattern = r'^Ang_X=\d+,Ang_Y=\d+$'
        return bool(re.match(pattern, payload.strip()))
    
    def _parse_angles(self, payload: str) -> tuple:
        """解析角度参数"""
        parts = payload.strip().split(',')
        
        # 解析X值
        x_part = parts[0].split('=')[1]
        ang_x = int(x_part)
        
        # 解析Y值
        y_part = parts[1].split('=')[1]
        ang_y = int(y_part)
        
        return ang_x, ang_y
    
    def _validate_angles(self, ang_x: int, ang_y: int) -> bool:
        """验证角度范围"""
        x_valid = self.position_limits['x']['min'] <= ang_x <= self.position_limits['x']['max']
        y_valid = self.position_limits['y']['min'] <= ang_y <= self.position_limits['y']['max']
        
        return x_valid and y_valid
    
    def _execute_control(self, ang_x: int, ang_y: int) -> bool:
        """
        执行云台控制
        模拟云台移动到指定位置
        
        Args:
            ang_x: X轴角度
            ang_y: Y轴角度
            
        Returns:
            执行是否成功
        """
        try:
            # 模拟云台移动时间（根据距离计算）
            old_pos = self.current_position.copy()
            
            # 计算移动距离
            x_distance = abs(ang_x - old_pos['x'])
            y_distance = abs(ang_y - old_pos['y'])
            total_distance = (x_distance ** 2 + y_distance ** 2) ** 0.5
            
            # 模拟移动时间（每100个单位需要0.1秒）
            move_time = max(0.1, total_distance / 1000)
            
            logger.info(f"云台开始移动: ({old_pos['x']}, {old_pos['y']}) -> ({ang_x}, {ang_y})")
            logger.info(f"预计移动时间: {move_time:.2f}秒")
            
            # 模拟移动延迟
            time.sleep(move_time)
            
            # 更新当前位置
            self.current_position = {"x": ang_x, "y": ang_y}
            
            logger.info(f"云台移动完成: 当前位置 ({ang_x}, {ang_y})")
            return True
            
        except Exception as e:
            logger.error(f"云台控制执行异常: {e}")
            return False
    
    def _register_device(self):
        """注册云台设备"""
        try:
            register_data = {
                'client_id': self.device_id,
                'username': self.username,
                'device_type': 'gimbal',
                'device_info': {
                    'model': 'Simulated Gimbal v1.0',
                    'position_limits': self.position_limits,
                    'current_position': self.current_position,
                    'capabilities': ['angle_control', 'position_feedback']
                }
            }
            
            self.client.publish(
                self.topics['register'], 
                json.dumps(register_data)
            )
            
            logger.info(f"云台设备注册信息已发送: {self.username} ({self.device_id})")
            
        except Exception as e:
            logger.error(f"注册云台设备异常: {e}")
    
    def _send_status_update(self):
        """发送状态更新"""
        try:
            status_data = {
                'client_id': self.device_id,
                'status': 'online',
                'current_position': self.current_position,
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy()
            }
            
            # 转换datetime对象为字符串
            if status_data['stats']['connect_time']:
                status_data['stats']['connect_time'] = status_data['stats']['connect_time'].isoformat()
            if status_data['stats']['last_command_time']:
                status_data['stats']['last_command_time'] = status_data['stats']['last_command_time'].isoformat()
            
            self.client.publish(
                self.topics['status'], 
                json.dumps(status_data)
            )
            
            logger.debug(f"云台状态更新已发送: {self.current_position}")
            
        except Exception as e:
            logger.error(f"发送状态更新异常: {e}")
    
    def _send_offline_status(self):
        """发送离线状态"""
        try:
            status_data = {
                'client_id': self.device_id,
                'status': 'offline',
                'timestamp': datetime.now().isoformat()
            }
            
            self.client.publish(
                self.topics['status'], 
                json.dumps(status_data)
            )
            
            logger.info("云台离线状态已发送")
            
        except Exception as e:
            logger.error(f"发送离线状态异常: {e}")
    
    def _send_chat_confirmation(self, ang_x: int, ang_y: int):
        """发送聊天确认消息（可选功能）"""
        try:
            # 这是一个可选功能，云台可以向聊天室发送确认消息
            chat_data = {
                'client_id': self.device_id,
                'username': f"{self.username} (设备)",
                'message': f"✅ 云台已移动到位置: X={ang_x}, Y={ang_y}"
            }
            
            self.client.publish(
                self.topics['chat_in'], 
                json.dumps(chat_data)
            )
            
            logger.debug(f"云台确认消息已发送: X={ang_x}, Y={ang_y}")
            
        except Exception as e:
            logger.error(f"发送聊天确认消息异常: {e}")
    
    def _start_status_timer(self):
        """启动状态发送定时器"""
        def send_periodic_status():
            if self.is_running and self.is_connected:
                self._send_status_update()
                # 安排下次发送
                self.status_timer = threading.Timer(30.0, send_periodic_status)
                self.status_timer.start()
        
        # 启动定时器
        self.status_timer = threading.Timer(30.0, send_periodic_status)
        self.status_timer.start()
        logger.info("状态发送定时器已启动 (30秒间隔)")
    
    def get_device_status(self) -> Dict[str, Any]:
        """获取设备状态信息"""
        return {
            'device_id': self.device_id,
            'username': self.username,
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'current_position': self.current_position,
            'position_limits': self.position_limits,
            'broker_info': f"{self.broker_host}:{self.broker_port}",
            'stats': self.stats.copy()
        }


def signal_handler(signum, frame):
    """信号处理器，用于优雅退出"""
    print(f"\n收到信号 {signum}，正在关闭云台设备...")
    if 'gimbal' in globals():
        gimbal.stop()
    sys.exit(0)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='云台设备模拟器')
    parser.add_argument('--host', default='localhost', help='MQTT代理服务器地址')
    parser.add_argument('--port', type=int, default=1883, help='MQTT代理服务器端口')
    parser.add_argument('--device-id', default=None, help='设备唯一标识符（留空则自动生成）')
    parser.add_argument('--username', default=None, help='MQTT用户名')
    parser.add_argument('--password', default=None, help='MQTT密码')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建云台设备实例
    global gimbal
    gimbal = GimbalDeviceSimulator(
        broker_host=args.host,
        broker_port=args.port,
        device_id=args.device_id
    )
    
    print(f"云台设备模拟器 v1.0")
    print(f"设备ID: {args.device_id}")
    print(f"MQTT代理: {args.host}:{args.port}")
    print("按 Ctrl+C 退出")
    print("-" * 50)
    
    # 启动云台设备
    if gimbal.start():
        try:
            # 主循环
            while gimbal.is_running:
                time.sleep(1)
                
                # 可以在这里添加其他周期性任务
                # 例如：监控系统状态、发送心跳等
                
        except KeyboardInterrupt:
            pass
    else:
        print("云台设备启动失败")
        sys.exit(1)
    
    # 清理退出
    gimbal.stop()
    print("云台设备模拟器已退出")


if __name__ == "__main__":
    main()
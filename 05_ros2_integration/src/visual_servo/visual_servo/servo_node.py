#!/usr/bin/env python3
"""
视觉伺服控制节点
基于视觉反馈的PID控制
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image
import numpy as np

class VisualServoNode(Node):
    def __init__(self):
        super().__init__('visual_servo')
        
        # 声明参数
        self.declare_parameter('target_class', 'apple')
        self.declare_parameter('kp_x', 0.5)
        self.declare_parameter('kp_y', 0.5)
        self.declare_parameter('ki_x', 0.01)
        self.declare_parameter('ki_y', 0.01)
        self.declare_parameter('kd_x', 0.1)
        self.declare_parameter('kd_y', 0.1)
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('target_tolerance', 10)
        
        # 获取参数
        self.target_class = self.get_parameter('target_class').value
        self.kp_x = self.get_parameter('kp_x').value
        self.kp_y = self.get_parameter('kp_y').value
        self.ki_x = self.get_parameter('ki_x').value
        self.ki_y = self.get_parameter('ki_y').value
        self.kd_x = self.get_parameter('kd_x').value
        self.kd_y = self.get_parameter('kd_y').value
        self.control_rate = self.get_parameter('control_rate').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.target_tolerance = self.get_parameter('target_tolerance').value
        
        # 图像中心
        self.center_x = self.image_width / 2
        self.center_y = self.image_height / 2
        
        # PID状态
        self.error_x = 0.0
        self.error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        
        # 当前目标
        self.current_target = None
        self.target_acquired = False
        
        # 订阅检测结果
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )
        
        # 发布控制命令
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # 发布目标位姿
        self.target_pose_pub = self.create_publisher(
            PoseStamped,
            '/target_pose',
            10
        )
        
        # 创建控制定时器
        control_period = 1.0 / self.control_rate
        self.control_timer = self.create_timer(
            control_period,
            self.control_callback
        )
        
        self.get_logger().info('视觉伺服节点已启动')
        self.get_logger().info(f'  目标类别: {self.target_class}')
        self.get_logger().info(f'  控制频率: {self.control_rate} Hz')
        self.get_logger().info(f'  PID参数: Kp=({self.kp_x}, {self.kp_y})')
    
    def detection_callback(self, msg):
        """检测结果回调"""
        # 查找目标物体
        target_found = False
        
        for detection in msg.detections:
            if len(detection.results) > 0:
                class_id = detection.results[0].hypothesis.class_id
                
                if class_id == self.target_class:
                    self.current_target = detection
                    target_found = True
                    
                    if not self.target_acquired:
                        self.target_acquired = True
                        self.get_logger().info(f'目标已锁定: {self.target_class}')
                    
                    break
        
        if not target_found:
            if self.target_acquired:
                self.get_logger().warn('目标丢失')
                self.target_acquired = False
            self.current_target = None
    
    def control_callback(self):
        """控制循环"""
        if not self.target_acquired or self.current_target is None:
            # 没有目标,停止运动
            self._publish_zero_velocity()
            return
        
        # 提取目标位置
        target_x = self.current_target.bbox.center.position.x
        target_y = self.current_target.bbox.center.position.y
        
        # 计算误差
        self.error_x = self.center_x - target_x
        self.error_y = self.center_y - target_y
        
        # 检查是否到达目标
        if (abs(self.error_x) < self.target_tolerance and
            abs(self.error_y) < self.target_tolerance):
            self.get_logger().info('目标已居中!')
            self._publish_zero_velocity()
            return
        
        # PID控制
        dt = 1.0 / self.control_rate
        
        # 积分项
        self.integral_x += self.error_x * dt
        self.integral_y += self.error_y * dt
        
        # 微分项
        derivative_x = (self.error_x - self.prev_error_x) / dt
        derivative_y = (self.error_y - self.prev_error_y) / dt
        
        # 计算控制输出
        vx = (self.kp_x * self.error_x +
              self.ki_x * self.integral_x +
              self.kd_x * derivative_x)
        
        vy = (self.kp_y * self.error_y +
              self.ki_y * self.integral_y +
              self.kd_y * derivative_y)
        
        # 更新历史
        self.prev_error_x = self.error_x
        self.prev_error_y = self.error_y
        
        # 发布控制命令
        cmd = Twist()
        cmd.linear.x = float(np.clip(vx / 1000.0, -0.5, 0.5))  # 归一化
        cmd.linear.y = float(np.clip(vy / 1000.0, -0.5, 0.5))
        self.cmd_pub.publish(cmd)
        
        # 调试信息
        if self.get_clock().now().nanoseconds % 1e9 < 2e7:  # 每秒打印一次
            self.get_logger().info(
                f'误差: ({self.error_x:.1f}, {self.error_y:.1f}), '
                f'速度: ({cmd.linear.x:.2f}, {cmd.linear.y:.2f})'
            )
    
    def _publish_zero_velocity(self):
        """发布零速度"""
        cmd = Twist()
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VisualServoNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'错误: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

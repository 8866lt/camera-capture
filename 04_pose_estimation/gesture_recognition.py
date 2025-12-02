#!/usr/bin/env python3
"""
手势识别系统
识别常见手势:挥手、举手、双手举起、指向等
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import argparse

class GestureRecognizer:
    def __init__(self):
        """初始化手势识别器"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 手势类别
        self.gestures = {
            'none': '无手势',
            'wave': '挥手',
            'raise_hand': '举手',
            'hands_up': '双手举起',
            'point_left': '指向左侧',
            'point_right': '指向右侧',
            'clap': '拍手',
            'arms_crossed': '双臂交叉'
        }
        
        # 用于检测挥手的历史
        self.wrist_x_history = deque(maxlen=30)
        
        print("手势识别器已初始化")
        print(f"支持的手势: {list(self.gestures.values())}")
    
    def calculate_angle(self, a, b, c):
        """
        计算三点夹角
        
        Args:
            a, b, c: 三个关键点
            
        Returns:
            angle: 角度(度)
        """
        a_pos = np.array([a.x, a.y])
        b_pos = np.array([b.x, b.y])
        c_pos = np.array([c.x, c.y])
        
        ba = a_pos - b_pos
        bc = c_pos - b_pos
        
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def is_arm_extended(self, shoulder, elbow, wrist):
        """判断手臂是否伸直"""
        angle = self.calculate_angle(shoulder, elbow, wrist)
        return angle > 150  # 角度>150°认为伸直
    
    def detect_wave(self, landmarks):
        """
        检测挥手动作
        通过检测手腕的左右摆动
        """
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # 记录手腕x坐标
        self.wrist_x_history.append(right_wrist.x)
        
        if len(self.wrist_x_history) < self.wrist_x_history.maxlen:
            return False
        
        # 计算方差(摆动幅度)
        x_array = np.array(list(self.wrist_x_history))
        variance = np.var(x_array)
        
        # 检测峰值数量(摆动次数)
        peaks = 0
        for i in range(1, len(x_array) - 1):
            if x_array[i] > x_array[i-1] and x_array[i] > x_array[i+1]:
                peaks += 1
        
        # 判断:方差大且有2-3个峰值
        if variance > 0.005 and 2 <= peaks <= 4:
            return True
        
        return False
    
    def recognize(self, landmarks):
        """
        识别手势
        
        Args:
            landmarks: MediaPipe关键点
            
        Returns:
            gesture: 手势类别
            confidence: 置信度
        """
        # 获取关键点
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # 特征提取
        features = {}
        
        # 1. 手腕相对于头部的位置
        features['left_wrist_above_head'] = left_wrist.y < nose.y
        features['right_wrist_above_head'] = right_wrist.y < nose.y
        
        # 2. 手腕相对于肩膀的位置
        features['left_wrist_above_shoulder'] = left_wrist.y < left_shoulder.y
        features['right_wrist_above_shoulder'] = right_wrist.y < right_shoulder.y
        
        # 3. 手臂是否伸直
        features['left_arm_extended'] = self.is_arm_extended(
            left_shoulder, left_elbow, left_wrist
        )
        features['right_arm_extended'] = self.is_arm_extended(
            right_shoulder, right_elbow, right_wrist
        )
        
        # 4. 手腕的水平位置
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        features['left_wrist_left_side'] = left_wrist.x < shoulder_center_x - 0.2
        features['right_wrist_right_side'] = right_wrist.x > shoulder_center_x + 0.2
        
        # 5. 手腕高度(相对于肩膀)
        features['right_wrist_shoulder_level'] = (
            abs(right_wrist.y - right_shoulder.y) < 0.1
        )
        
        # 规则匹配
        gesture, confidence = self._match_gesture(features, landmarks)
        
        return gesture, confidence, features
    
    def _match_gesture(self, features, landmarks):
        """基于规则匹配手势"""
        
        # 双手举起
        if (features['left_wrist_above_head'] and 
            features['right_wrist_above_head']):
            return 'hands_up', 0.95
        
        # 举手(右手)
        if (features['right_wrist_above_shoulder'] and 
            features['right_arm_extended'] and
            not features['right_wrist_above_head']):
            return 'raise_hand', 0.9
        
        # 挥手
        if (features['right_wrist_shoulder_level'] and
            self.detect_wave(landmarks)):
            return 'wave', 0.85
        
        # 指向左侧
        if (features['right_arm_extended'] and
            features['right_wrist_shoulder_level'] and
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x < 0.3):
            return 'point_left', 0.8
        
        # 指向右侧
        if (features['right_arm_extended'] and
            features['right_wrist_shoulder_level'] and
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x > 0.7):
            return 'point_right', 0.8
        
        # 双臂交叉
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        if (left_wrist.x > right_wrist.x and
            abs(left_wrist.y - right_wrist.y) < 0.15):
            return 'arms_crossed', 0.75
        
        return 'none', 0.0
    
    def detect_and_recognize(self, frame):
        """
        检测姿态并识别手势
        
        Args:
            frame: 输入图像
            
        Returns:
            gesture: 手势类别
            confidence: 置信度
            annotated: 标注后的图像
        """
        # 转换颜色
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 姿态检测
        results = self.pose.process(rgb)
        
        annotated = frame.copy()
        gesture = 'none'
        confidence = 0.0
        
        if results.pose_landmarks:
            # 识别手势
            gesture, confidence, features = self.recognize(
                results.pose_landmarks.landmark
            )
            
            # 绘制骨架
            self.mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # 显示手势
            h, w = frame.shape[:2]
            
            gesture_text = self.gestures.get(gesture, gesture)
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            
            cv2.putText(annotated, f"Gesture: {gesture_text}",
                       (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            cv2.putText(annotated, f"Confidence: {confidence:.2f}",
                       (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # 显示特征(调试用)
            if gesture != 'none':
                y = 130
                for key, value in features.items():
                    if value:
                        cv2.putText(annotated, f"  {key}",
                                   (10, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                   (200, 200, 200), 1)
                        y += 25
        
        else:
            cv2.putText(annotated, "No person detected", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return gesture, confidence, annotated
    
    def release(self):
        """释放资源"""
        self.pose.close()

def main():
    parser = argparse.ArgumentParser(description='手势识别系统')
    parser.add_argument('--source', type=str, default='0',
                       help='输入源')
    
    args = parser.parse_args()
    
    # 创建识别器
    recognizer = GestureRecognizer()
    
    # 打开输入源
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"无法打开输入源: {source}")
        return
    
    print("\n手势识别系统已启动")
    print("尝试做出以下手势:")
    for gesture_name in recognizer.gestures.values():
        print(f"  - {gesture_name}")
    print("\n按 'q' 退出\n")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 识别手势
            gesture, confidence, annotated = recognizer.detect_and_recognize(frame)
            
            # 显示
            cv2.imshow('Gesture Recognition', annotated)
            
            # 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.release()

if __name__ == '__main__':
    main()

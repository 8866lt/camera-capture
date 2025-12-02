#!/usr/bin/env python3
"""
相机标定 - 使用棋盘格标定板
原理:通过拍摄多张不同角度的棋盘格图案,计算相机内参和畸变系数
"""

import cv2
import numpy as np
import glob
import os
import yaml
from pathlib import Path

class CameraCalibration:
    def __init__(self, chessboard_size=(9, 6), square_size=0.025):
        """
        初始化标定器
        
        Args:
            chessboard_size: 棋盘格内角点数量 (列数, 行数)
                           注意:是内角点,不是格子数
                           9x6表示10x7个格子
            square_size: 棋盘格大小(米),例如25mm = 0.025m
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # 准备3D点坐标 (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), 
                             np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 
                                     0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size  # 转换为实际物理尺寸
        
        # 存储所有图像的角点
        self.obj_points = []  # 3D点
        self.img_points = []  # 2D点
        
        # 标定结果
        self.camera_matrix = None  # 内参矩阵
        self.dist_coeffs = None    # 畸变系数
        self.rvecs = None          # 旋转向量
        self.tvecs = None          # 平移向量
        self.img_size = None       # 图像尺寸
        
    def find_corners(self, image, show=False):
        """
        在图像中查找棋盘格角点
        
        Args:
            image: 输入图像(BGR)
            show: 是否显示检测结果
            
        Returns:
            success: 是否找到角点
            corners: 角点坐标
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        # 使用自适应阈值和归一化有助于提高检测成功率
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # 亚像素精化
            # 这一步可以将角点定位精度从1像素提升到0.1像素
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                       30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                       criteria)
            
            if show:
                # 绘制角点
                img_with_corners = image.copy()
                cv2.drawChessboardCorners(img_with_corners, 
                                         self.chessboard_size, 
                                         corners, ret)
                cv2.imshow('Chessboard', img_with_corners)
                cv2.waitKey(500)
        
        return ret, corners
    
    def add_image(self, image):
        """
        添加一张标定图像
        
        Args:
            image: 输入图像
            
        Returns:
            success: 是否成功添加
        """
        ret, corners = self.find_corners(image)
        
        if ret:
            self.obj_points.append(self.objp)
            self.img_points.append(corners)
            
            if self.img_size is None:
                self.img_size = (image.shape[1], image.shape[0])
            
            print(f"成功添加第 {len(self.obj_points)} 张图像")
            return True
        else:
            print("未检测到棋盘格")
            return False
    
    def calibrate(self):
        """
        执行标定计算
        
        Returns:
            success: 标定是否成功
            rms_error: 重投影误差(像素)
        """
        if len(self.obj_points) < 10:
            print(f"警告: 只有 {len(self.obj_points)} 张图像,建议至少10张")
        
        print(f"\n开始标定,使用 {len(self.obj_points)} 张图像...")
        
        # 执行标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points,
            self.img_points,
            self.img_size,
            None,
            None
        )
        
        if ret:
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            
            print(f"\n标定成功!")
            print(f"重投影误差(RMS): {ret:.4f} 像素")
            print(f"\n相机内参矩阵:")
            print(mtx)
            print(f"\n畸变系数:")
            print(dist)
            
            # 计算FOV
            fx, fy = mtx[0, 0], mtx[1, 1]
            cx, cy = mtx[0, 2], mtx[1, 2]
            fov_x = 2 * np.arctan(self.img_size[0] / (2 * fx)) * 180 / np.pi
            fov_y = 2 * np.arctan(self.img_size[1] / (2 * fy)) * 180 / np.pi
            print(f"\n视场角(FOV):")
            print(f"  水平: {fov_x:.2f}°")
            print(f"  垂直: {fov_y:.2f}°")
            
            return True, ret
        else:
            print("标定失败!")
            return False, 0
    
    def save(self, filename='camera_calibration.yaml'):
        """保存标定结果到YAML文件"""
        if self.camera_matrix is None:
            print("尚未标定,无法保存")
            return
        
        data = {
            'image_width': int(self.img_size[0]),
            'image_height': int(self.img_size[1]),
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.dist_coeffs.tolist(),
            'chessboard_size': list(self.chessboard_size),
            'square_size': float(self.square_size)
        }
        
        with open(filename, 'w') as f:
            yaml.dump(data, f)
        
        print(f"标定结果已保存到: {filename}")
    
    @staticmethod
    def load(filename='camera_calibration.yaml'):
        """从YAML文件加载标定结果"""
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        
        calib = CameraCalibration()
        calib.img_size = (data['image_width'], data['image_height'])
        calib.camera_matrix = np.array(data['camera_matrix'])
        calib.dist_coeffs = np.array(data['distortion_coefficients'])
        
        return calib
    
    def undistort(self, image):
        """
        对图像进行畸变校正
        
        Args:
            image: 输入畸变图像
            
        Returns:
            undistorted: 校正后的图像
        """
        if self.camera_matrix is None:
            raise RuntimeError("尚未标定或加载标定参数")
        
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='相机标定')
    parser.add_argument('--mode', choices=['capture', 'calibrate', 'test'],
                       required=True, help='运行模式')
    parser.add_argument('--camera', type=int, default=0,
                       help='相机ID')
    parser.add_argument('--images', type=str, default='calib_images/*.jpg',
                       help='标定图像路径(calibrate模式)')
    parser.add_argument('--output', type=str, default='camera_calibration.yaml',
                       help='输出文件名')
    parser.add_argument('--cols', type=int, default=9,
                       help='棋盘格列数(内角点)')
    parser.add_argument('--rows', type=int, default=6,
                       help='棋盘格行数(内角点)')
    parser.add_argument('--size', type=float, default=0.025,
                       help='棋盘格大小(米)')
    args = parser.parse_args()
    
    calibrator = CameraCalibration(
        chessboard_size=(args.cols, args.rows),
        square_size=args.size
    )
    
    if args.mode == 'capture':
        # 采集模式:拍摄标定图像
        print("采集模式 - 按's'保存当前帧,'q'退出")
        
        os.makedirs('calib_images', exist_ok=True)
        cap = cv2.VideoCapture(args.camera)
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 实时检测棋盘格
            success, corners = calibrator.find_corners(frame)
            
            if success:
                # 绘制角点
                img_show = frame.copy()
                cv2.drawChessboardCorners(img_show, 
                                         calibrator.chessboard_size,
                                         corners, True)
                cv2.putText(img_show, "Detected! Press 's' to save", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                cv2.imshow('Calibration', img_show)
            else:
                cv2.putText(frame, "Not detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and success:
                filename = f'calib_images/calib_{count:03d}.jpg'
                cv2.imwrite(filename, frame)
                print(f"已保存: {filename}")
                count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n共采集 {count} 张图像")
    
    elif args.mode == 'calibrate':
        # 标定模式:使用已有图像进行标定
        print("标定模式")
        
        images = glob.glob(args.images)
        print(f"找到 {len(images)} 张图像")
        
        for fname in images:
            img = cv2.imread(fname)
            calibrator.add_image(img)
        
        if len(calibrator.obj_points) > 0:
            success, error = calibrator.calibrate()
            if success:
                calibrator.save(args.output)
        else:
            print("未找到有效的标定图像")
    
    elif args.mode == 'test':
        # 测试模式:实时显示畸变校正效果
        print("测试模式 - 实时显示畸变校正效果")
        
        calibrator = CameraCalibration.load(args.output)
        cap = cv2.VideoCapture(args.camera)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 畸变校正
            undistorted = calibrator.undistort(frame)
            
            # 并排显示
            combined = np.hstack([frame, undistorted])
            cv2.putText(combined, "Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Undistorted", 
                       (frame.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Calibration Test', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

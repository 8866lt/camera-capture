#!/usr/bin/env python3
"""
标定质量检查工具
检查标定结果的质量,分析潜在问题,给出改进建议
"""

import cv2
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import glob

class CalibrationChecker:
    def __init__(self, calib_file):
        """
        加载标定结果
        
        Args:
            calib_file: 标定参数YAML文件
        """
        with open(calib_file, 'r') as f:
            data = yaml.safe_load(f)
        
        self.image_width = data['image_width']
        self.image_height = data['image_height']
        self.camera_matrix = np.array(data['camera_matrix'])
        self.dist_coeffs = np.array(data['distortion_coefficients']).flatten()
        
        # 提取参数
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        
        self.k1, self.k2, self.p1, self.p2, self.k3 = self.dist_coeffs
        
        print(f"标定参数已加载: {calib_file}")
        print(f"  图像尺寸: {self.image_width} x {self.image_height}")
    
    def check_intrinsics(self):
        """检查内参矩阵是否合理"""
        print("\n===== 内参矩阵检查 =====")
        
        issues = []
        warnings = []
        
        # 检查1: 焦距合理性
        fx_ratio = self.fx / self.image_width
        fy_ratio = self.fy / self.image_height
        
        print(f"焦距:")
        print(f"  fx = {self.fx:.2f} ({fx_ratio:.2f} × 图像宽度)")
        print(f"  fy = {self.fy:.2f} ({fy_ratio:.2f} × 图像高度)")
        
        if fx_ratio < 0.3 or fx_ratio > 3.0:
            issues.append(f"焦距异常: fx/width = {fx_ratio:.2f} (正常范围0.5-2.0)")
        
        if abs(fx_ratio - fy_ratio) > 0.1:
            warnings.append(f"焦距不对称: fx/fy = {self.fx/self.fy:.3f} (通常应接近1.0)")
        
        # 检查2: 主点位置
        cx_offset = abs(self.cx - self.image_width / 2)
        cy_offset = abs(self.cy - self.image_height / 2)
        
        print(f"\n主点:")
        print(f"  cx = {self.cx:.2f} (图像中心: {self.image_width/2:.1f}, 偏移: {cx_offset:.1f})")
        print(f"  cy = {self.cy:.2f} (图像中心: {self.image_height/2:.1f}, 偏移: {cy_offset:.1f})")
        
        if cx_offset > self.image_width * 0.1:
            warnings.append(f"主点x偏移较大: {cx_offset:.1f} 像素")
        
        if cy_offset > self.image_height * 0.1:
            warnings.append(f"主点y偏移较大: {cy_offset:.1f} 像素")
        
        # 检查3: 计算视场角
        fov_x = 2 * np.arctan(self.image_width / (2 * self.fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(self.image_height / (2 * self.fy)) * 180 / np.pi
        
        print(f"\n视场角:")
        print(f"  水平FOV: {fov_x:.1f}°")
        print(f"  垂直FOV: {fov_y:.1f}°")
        
        if fov_x > 120:
            print(f"  → 广角镜头 (可能有较大畸变)")
        elif fov_x > 70:
            print(f"  → 标准镜头")
        else:
            print(f"  → 长焦镜头")
        
        return issues, warnings
    
    def check_distortion(self):
        """检查畸变系数是否合理"""
        print("\n===== 畸变系数检查 =====")
        
        issues = []
        warnings = []
        
        print(f"径向畸变:")
        print(f"  k1 = {self.k1:.6f}")
        print(f"  k2 = {self.k2:.6f}")
        print(f"  k3 = {self.k3:.6f}")
        
        print(f"\n切向畸变:")
        print(f"  p1 = {self.p1:.6f}")
        print(f"  p2 = {self.p2:.6f}")
        
        # 判断畸变类型
        if abs(self.k1) > 0.5:
            issues.append(f"径向畸变极大: k1={self.k1:.3f} (可能是鱼眼镜头)")
        elif abs(self.k1) > 0.3:
            warnings.append(f"径向畸变较大: k1={self.k1:.3f} (广角镜头)")
        
        if self.k1 > 0:
            print("  → 桶形畸变 (边缘向外凸)")
        elif self.k1 < 0:
            print("  → 枕形畸变 (边缘向内凹)")
        else:
            print("  → 无明显畸变")
        
        if abs(self.p1) > 0.01 or abs(self.p2) > 0.01:
            warnings.append(f"切向畸变较大: p1={self.p1:.4f}, p2={self.p2:.4f}")
        
        return issues, warnings
    
    def visualize_distortion(self, output_file='distortion_map.png'):
        """可视化畸变分布"""
        print(f"\n生成畸变可视化图: {output_file}")
        
        # 创建网格
        grid_size = 20
        x = np.linspace(0, self.image_width, grid_size)
        y = np.linspace(0, self.image_height, grid_size)
        xv, yv = np.meshgrid(x, y)
        
        # 转为归一化坐标
        points = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float32)
        points = points.reshape(-1, 1, 2)
        
        # 去畸变
        points_undist = cv2.undistortPoints(
            points, 
            self.camera_matrix, 
            self.dist_coeffs,
            None,
            self.camera_matrix
        )
        
        # 计算畸变向量
        points = points.reshape(-1, 2)
        points_undist = points_undist.reshape(-1, 2)
        distortion_vectors = points_undist - points
        
        # 计算畸变大小
        distortion_magnitude = np.linalg.norm(distortion_vectors, axis=1)
        distortion_magnitude = distortion_magnitude.reshape(grid_size, grid_size)
        
        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图: 畸变向量场
        ax = axes[0]
        xv_grid = xv[::2, ::2]
        yv_grid = yv[::2, ::2]
        u = distortion_vectors[::2*grid_size].reshape(grid_size//2, grid_size//2, 2)[:, :, 0]
        v = distortion_vectors[::2*grid_size].reshape(grid_size//2, grid_size//2, 2)[:, :, 1]
        
        ax.quiver(xv_grid, yv_grid, u, v, scale=50, color='red', alpha=0.7)
        ax.set_xlim(0, self.image_width)
        ax.set_ylim(self.image_height, 0)
        ax.set_aspect('equal')
        ax.set_title('畸变向量场\n(箭头表示校正时像素移动方向)', fontsize=12)
        ax.set_xlabel('X (像素)')
        ax.set_ylabel('Y (像素)')
        ax.grid(True, alpha=0.3)
        
        # 右图: 畸变大小热力图
        ax = axes[1]
        im = ax.imshow(distortion_magnitude, extent=[0, self.image_width, self.image_height, 0],
                      cmap='hot', aspect='auto')
        ax.set_title('畸变大小分布\n(颜色越亮畸变越大)', fontsize=12)
        ax.set_xlabel('X (像素)')
        ax.set_ylabel('Y (像素)')
        plt.colorbar(im, ax=ax, label='畸变大小(像素)')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  最大畸变: {distortion_magnitude.max():.2f} 像素")
        print(f"  平均畸变: {distortion_magnitude.mean():.2f} 像素")
        
        return distortion_magnitude.max()
    
    def check_calibration_images(self, image_pattern, chessboard_size):
        """检查标定图像质量"""
        print(f"\n===== 标定图像检查 =====")
        
        images = glob.glob(image_pattern)
        if len(images) == 0:
            print(f"未找到图像: {image_pattern}")
            return
        
        print(f"找到 {len(images)} 张图像")
        
        # 准备3D点
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 
                              0:chessboard_size[1]].T.reshape(-1, 2)
        
        objpoints = []
        imgpoints = []
        
        detection_count = 0
        reprojection_errors = []
        
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size)
            
            if ret:
                detection_count += 1
                
                # 亚像素精化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                
                objpoints.append(objp)
                imgpoints.append(corners)
        
        print(f"成功检测: {detection_count}/{len(images)} ({detection_count/len(images)*100:.1f}%)")
        
        if detection_count < 10:
            print("警告: 有效图像太少(<10张),建议重新采集")
            return
        
        # 计算每张图的重投影误差
        print("\n重投影误差分析:")
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                objpoints[i], 
                np.zeros(3), np.zeros(3),  # 假设外参
                self.camera_matrix, 
                self.dist_coeffs
            )
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            reprojection_errors.append(error)
        
        errors = np.array(reprojection_errors)
        print(f"  平均误差: {errors.mean():.4f} 像素")
        print(f"  最大误差: {errors.max():.4f} 像素")
        print(f"  最小误差: {errors.min():.4f} 像素")
        print(f"  标准差: {errors.std():.4f} 像素")
        
        # 找出误差大的图像
        threshold = errors.mean() + 2 * errors.std()
        outliers = np.where(errors > threshold)[0]
        
        if len(outliers) > 0:
            print(f"\n检测到 {len(outliers)} 张可能有问题的图像(误差>{threshold:.3f}):")
            for idx in outliers:
                print(f"  {Path(images[idx]).name}: {errors[idx]:.4f} 像素")
            print("  建议检查这些图像是否模糊或角点检测错误")
    
    def test_undistortion(self, test_image, output_file='undistortion_test.jpg'):
        """测试畸变校正效果"""
        print(f"\n===== 畸变校正测试 =====")
        
        img = cv2.imread(test_image)
        if img is None:
            print(f"无法读取图像: {test_image}")
            return
        
        # 计算最优新相机矩阵
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        # 校正
        dst = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, 
                           None, newcameramtx)
        
        # 裁剪ROI
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            dst_cropped = dst[y:y+h_roi, x:x+w_roi]
            print(f"校正后ROI: {w_roi} x {h_roi}")
            print(f"裁剪比例: {w_roi/w*100:.1f}% x {h_roi/h*100:.1f}%")
        else:
            dst_cropped = dst
        
        # 拼接对比图
        h_min = min(img.shape[0], dst_cropped.shape[0])
        img_resized = cv2.resize(img, (int(img.shape[1]*h_min/img.shape[0]), h_min))
        dst_resized = cv2.resize(dst_cropped, 
                                (int(dst_cropped.shape[1]*h_min/dst_cropped.shape[0]), h_min))
        
        # 添加标签
        cv2.putText(img_resized, "Original", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(dst_resized, "Undistorted", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        comparison = np.hstack([img_resized, dst_resized])
        cv2.imwrite(output_file, comparison)
        
        print(f"对比图已保存: {output_file}")
        print("  观察直线(门框、墙角)是否变直")
    
    def generate_report(self):
        """生成完整检查报告"""
        print("\n" + "="*60)
        print("标定质量报告")
        print("="*60)
        
        all_issues = []
        all_warnings = []
        
        # 检查内参
        issues, warnings = self.check_intrinsics()
        all_issues.extend(issues)
        all_warnings.extend(warnings)
        
        # 检查畸变
        issues, warnings = self.check_distortion()
        all_issues.extend(issues)
        all_warnings.extend(warnings)
        
        # 可视化畸变
        max_distortion = self.visualize_distortion()
        
        # 总结
        print("\n" + "="*60)
        print("检查总结")
        print("="*60)
        
        if len(all_issues) == 0 and len(all_warnings) == 0:
            print("✓ 标定参数看起来合理")
        else:
            if len(all_issues) > 0:
                print(f"\n❌ 发现 {len(all_issues)} 个严重问题:")
                for issue in all_issues:
                    print(f"  - {issue}")
                print("\n建议: 重新标定")
            
            if len(all_warnings) > 0:
                print(f"\n⚠ 发现 {len(all_warnings)} 个警告:")
                for warning in all_warnings:
                    print(f"  - {warning}")
                print("\n建议: 检查但可以继续使用")
        
        # 质量评级
        print("\n总体质量评级:")
        if len(all_issues) > 0:
            print("  ⭐ 较差 - 需要重新标定")
        elif len(all_warnings) > 2:
            print("  ⭐⭐ 一般 - 建议改进")
        elif len(all_warnings) > 0:
            print("  ⭐⭐⭐ 良好 - 可以使用")
        else:
            print("  ⭐⭐⭐⭐ 优秀 - 质量很好")

def main():
    parser = argparse.ArgumentParser(description='标定质量检查工具')
    parser.add_argument('--calib', type=str, required=True,
                       help='标定参数文件(YAML)')
    parser.add_argument('--images', type=str,
                       help='标定图像路径(用于重投影误差分析)')
    parser.add_argument('--chessboard', type=str, default='9,6',
                       help='棋盘格尺寸 "cols,rows"')
    parser.add_argument('--test-image', type=str,
                       help='测试畸变校正的图像')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 解析棋盘格尺寸
    cols, rows = map(int, args.chessboard.split(','))
    chessboard_size = (cols, rows)
    
    # 创建检查器
    checker = CalibrationChecker(args.calib)
    
    # 生成报告
    checker.generate_report()
    
    # 检查标定图像
    if args.images:
        checker.check_calibration_images(args.images, chessboard_size)
    
    # 测试畸变校正
    if args.test_image:
        output_file = Path(args.output_dir) / 'undistortion_test.jpg'
        checker.test_undistortion(args.test_image, str(output_file))
    
    print("\n检查完成!")

if __name__ == '__main__':
    main()

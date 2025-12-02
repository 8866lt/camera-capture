#!/usr/bin/env python3
"""
生成各种标定板图案
支持: 棋盘格、ChArUco板、圆点阵、非对称圆点阵
可以直接保存为PDF打印
"""

import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import argparse

class CalibrationPatternGenerator:
    def __init__(self, page_size='A4'):
        """
        初始化标定板生成器
        
        Args:
            page_size: 纸张尺寸 'A4' 或 'Letter'
        """
        if page_size == 'A4':
            self.page_width, self.page_height = A4
        else:  # Letter
            self.page_width = 8.5 * 25.4 * mm
            self.page_height = 11 * 25.4 * mm
    
    def generate_chessboard(self, rows, cols, square_size, output_file='chessboard.pdf'):
        """
        生成棋盘格标定板
        
        Args:
            rows: 行数(内角点数)
            cols: 列数(内角点数)
            square_size: 格子尺寸(mm)
            output_file: 输出PDF文件
        """
        # 计算总尺寸
        pattern_width = (cols + 1) * square_size * mm
        pattern_height = (rows + 1) * square_size * mm
        
        # 检查是否超出纸张
        if pattern_width > self.page_width * 0.9 or pattern_height > self.page_height * 0.9:
            print(f"警告: 图案尺寸({pattern_width/mm:.1f}x{pattern_height/mm:.1f}mm) "
                  f"可能超出纸张({self.page_width/mm:.1f}x{self.page_height/mm:.1f}mm)")
        
        # 创建PDF
        c = canvas.Canvas(output_file, pagesize=(self.page_width, self.page_height))
        
        # 居中位置
        x_offset = (self.page_width - pattern_width) / 2
        y_offset = (self.page_height - pattern_height) / 2
        
        # 绘制标题和信息
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(self.page_width/2, self.page_height - 30*mm, 
                           "Camera Calibration Chessboard")
        
        c.setFont("Helvetica", 10)
        info_y = self.page_height - 45*mm
        c.drawString(30*mm, info_y, f"Pattern: {cols+1} x {rows+1} squares")
        c.drawString(30*mm, info_y - 5*mm, f"Inner corners: {cols} x {rows}")
        c.drawString(30*mm, info_y - 10*mm, f"Square size: {square_size} mm")
        c.drawString(30*mm, info_y - 15*mm, f"Total size: {(cols+1)*square_size:.1f} x {(rows+1)*square_size:.1f} mm")
        
        c.drawString(30*mm, info_y - 25*mm, "Print at 100% scale (no fit to page)")
        c.drawString(30*mm, info_y - 30*mm, "Verify with ruler after printing")
        
        # 绘制棋盘格
        for i in range(rows + 1):
            for j in range(cols + 1):
                x = x_offset + j * square_size * mm
                y = y_offset + i * square_size * mm
                
                # 棋盘格着色规则
                if (i + j) % 2 == 0:
                    c.setFillColorRGB(0, 0, 0)  # 黑色
                else:
                    c.setFillColorRGB(1, 1, 1)  # 白色
                
                c.rect(x, y, square_size * mm, square_size * mm, fill=1, stroke=0)
        
        # 绘制外框(用于验证尺寸)
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(0.5)
        c.rect(x_offset, y_offset, pattern_width, pattern_height, fill=0, stroke=1)
        
        # 添加刻度标记(便于验证尺寸)
        c.setFont("Helvetica", 6)
        for i in range(0, cols + 2, 2):
            x = x_offset + i * square_size * mm
            c.line(x, y_offset - 2*mm, x, y_offset - 5*mm)
            c.drawCentredString(x, y_offset - 8*mm, f"{i*square_size:.0f}")
        
        for i in range(0, rows + 2, 2):
            y = y_offset + i * square_size * mm
            c.line(x_offset - 2*mm, y, x_offset - 5*mm, y)
            c.drawRightString(x_offset - 6*mm, y - 2*mm, f"{i*square_size:.0f}")
        
        c.drawString(x_offset - 6*mm, y_offset - 12*mm, "mm")
        
        c.save()
        print(f"棋盘格已生成: {output_file}")
        print(f"  内角点: {cols} x {rows}")
        print(f"  格子数: {cols+1} x {rows+1}")
        print(f"  格子尺寸: {square_size} mm")
        print(f"  总尺寸: {(cols+1)*square_size:.1f} x {(rows+1)*square_size:.1f} mm")
    
    def generate_charuco(self, rows, cols, square_size, marker_size, 
                        dict_type=cv2.aruco.DICT_6X6_250, output_file='charuco.pdf'):
        """
        生成ChArUco标定板
        
        Args:
            rows: 行数(内角点数)
            cols: 列数(内角点数)
            square_size: 格子尺寸(mm)
            marker_size: ArUco标记尺寸(mm),通常是square_size的80%
            dict_type: ArUco字典类型
            output_file: 输出PDF文件
        """
        # 创建ChArUco板
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        board = cv2.aruco.CharucoBoard(
            (cols+1, rows+1),
            square_size / 1000,  # 转为米
            marker_size / 1000,
            aruco_dict
        )
        
        # 生成图像(高分辨率)
        dpi = 300
        img_width = int((cols + 1) * square_size / 25.4 * dpi)
        img_height = int((rows + 1) * square_size / 25.4 * dpi)
        
        board_image = board.generateImage((img_width, img_height), marginSize=0)
        
        # 保存为临时PNG
        temp_png = output_file.replace('.pdf', '_temp.png')
        cv2.imwrite(temp_png, board_image)
        
        # 创建PDF并嵌入图像
        c = canvas.Canvas(output_file, pagesize=(self.page_width, self.page_height))
        
        # 计算显示尺寸
        pattern_width = (cols + 1) * square_size * mm
        pattern_height = (rows + 1) * square_size * mm
        
        x_offset = (self.page_width - pattern_width) / 2
        y_offset = (self.page_height - pattern_height) / 2
        
        # 标题
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(self.page_width/2, self.page_height - 30*mm,
                           "ChArUco Calibration Board")
        
        c.setFont("Helvetica", 10)
        info_y = self.page_height - 45*mm
        c.drawString(30*mm, info_y, f"Pattern: {cols+1} x {rows+1} squares")
        c.drawString(30*mm, info_y - 5*mm, f"Inner corners: {cols} x {rows}")
        c.drawString(30*mm, info_y - 10*mm, f"Square size: {square_size} mm")
        c.drawString(30*mm, info_y - 15*mm, f"Marker size: {marker_size} mm")
        
        # 嵌入图像
        c.drawImage(temp_png, x_offset, y_offset, 
                   width=pattern_width, height=pattern_height)
        
        c.save()
        
        # 删除临时文件
        import os
        os.remove(temp_png)
        
        print(f"ChArUco板已生成: {output_file}")
    
    def generate_circles(self, rows, cols, circle_spacing, circle_diameter, 
                        asymmetric=False, output_file='circles.pdf'):
        """
        生成圆点阵标定板
        
        Args:
            rows: 行数
            cols: 列数
            circle_spacing: 圆心距(mm)
            circle_diameter: 圆直径(mm)
            asymmetric: 是否为非对称圆点阵
            output_file: 输出PDF文件
        """
        # 创建PDF
        c = canvas.Canvas(output_file, pagesize=(self.page_width, self.page_height))
        
        # 计算总尺寸
        if asymmetric:
            pattern_width = cols * circle_spacing * mm
            pattern_height = (rows - 0.5) * circle_spacing * mm
        else:
            pattern_width = (cols - 1) * circle_spacing * mm
            pattern_height = (rows - 1) * circle_spacing * mm
        
        x_offset = (self.page_width - pattern_width) / 2
        y_offset = (self.page_height - pattern_height) / 2
        
        # 标题
        c.setFont("Helvetica-Bold", 14)
        title = "Asymmetric Circles" if asymmetric else "Symmetric Circles"
        c.drawCentredString(self.page_width/2, self.page_height - 30*mm,
                           f"{title} Calibration Pattern")
        
        c.setFont("Helvetica", 10)
        info_y = self.page_height - 45*mm
        c.drawString(30*mm, info_y, f"Pattern: {cols} x {rows} circles")
        c.drawString(30*mm, info_y - 5*mm, f"Circle spacing: {circle_spacing} mm")
        c.drawString(30*mm, info_y - 10*mm, f"Circle diameter: {circle_diameter} mm")
        
        # 绘制圆点
        c.setFillColorRGB(0, 0, 0)
        radius = circle_diameter / 2 * mm
        
        for i in range(rows):
            for j in range(cols):
                if asymmetric and i % 2 == 1:
                    x = x_offset + (j + 0.5) * circle_spacing * mm
                else:
                    x = x_offset + j * circle_spacing * mm
                
                y = y_offset + i * circle_spacing * mm
                
                c.circle(x, y, radius, fill=1, stroke=0)
        
        c.save()
        print(f"圆点阵已生成: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='生成相机标定板',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成9x6棋盘格,25mm格子
  python generate_pattern.py --type chessboard --rows 6 --cols 9 --size 25
  
  # 生成ChArUco板
  python generate_pattern.py --type charuco --rows 6 --cols 9 --size 25 --marker-size 20
  
  # 生成圆点阵
  python generate_pattern.py --type circles --rows 7 --cols 11 --spacing 20 --diameter 10
        """
    )
    
    parser.add_argument('--type', type=str, required=True,
                       choices=['chessboard', 'charuco', 'circles', 'circles_asymmetric'],
                       help='标定板类型')
    parser.add_argument('--rows', type=int, required=True,
                       help='行数(对于棋盘格是内角点数)')
    parser.add_argument('--cols', type=int, required=True,
                       help='列数(对于棋盘格是内角点数)')
    parser.add_argument('--size', type=float, default=25,
                       help='格子尺寸(mm),用于chessboard和charuco')
    parser.add_argument('--marker-size', type=float, default=None,
                       help='ArUco标记尺寸(mm),用于charuco,默认为size*0.8')
    parser.add_argument('--spacing', type=float, default=20,
                       help='圆心距(mm),用于circles')
    parser.add_argument('--diameter', type=float, default=10,
                       help='圆直径(mm),用于circles')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件名,默认为<type>.pdf')
    parser.add_argument('--page-size', type=str, default='A4',
                       choices=['A4', 'Letter'],
                       help='纸张尺寸')
    
    args = parser.parse_args()
    
    # 默认输出文件名
    if args.output is None:
        args.output = f"{args.type}.pdf"
    
    # 创建生成器
    generator = CalibrationPatternGenerator(page_size=args.page_size)
    
    # 生成标定板
    if args.type == 'chessboard':
        generator.generate_chessboard(args.rows, args.cols, args.size, args.output)
    
    elif args.type == 'charuco':
        marker_size = args.marker_size if args.marker_size else args.size * 0.8
        generator.generate_charuco(args.rows, args.cols, args.size, 
                                  marker_size, output_file=args.output)
    
    elif args.type == 'circles':
        generator.generate_circles(args.rows, args.cols, args.spacing, 
                                  args.diameter, asymmetric=False, 
                                  output_file=args.output)
    
    elif args.type == 'circles_asymmetric':
        generator.generate_circles(args.rows, args.cols, args.spacing,
                                  args.diameter, asymmetric=True,
                                  output_file=args.output)
    
    print(f"\n✓ 标定板已生成: {args.output}")
    print("\n打印说明:")
    print("  1. 使用激光打印机(喷墨会渗色)")
    print("  2. 打印设置选择'实际尺寸'或'100%',不要'适应页面'")
    print("  3. 打印后用尺子验证实际尺寸")
    print("  4. 粘贴到硬纸板或亚克力板上保持平整")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
YOLO物体检测 - 基于ultralytics
支持: YOLOv8, YOLOv5
支持导出: ONNX, TensorRT, OpenVINO
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
from pathlib import Path

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', conf_thresh=0.5, iou_thresh=0.4, device='cpu'):
        """
        初始化YOLO检测器
        
        Args:
            model_path: 模型路径
                - 预训练模型: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'
                - TensorRT引擎: 'yolov8n.engine'
                - ONNX模型: 'yolov8n.onnx'
            conf_thresh: 置信度阈值
            iou_thresh: NMS IOU阈值
            device: 'cpu', 'cuda:0', 'cuda:1', ...
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 设置设备
        self.model.to(device)
        
        # 如果模型不存在,会自动下载
        # yolov8n.pt: ~6MB, 最快但精度最低
        # yolov8s.pt: ~22MB, 平衡
        # yolov8m.pt: ~52MB, 精度较高
        # yolov8l.pt: ~87MB, 高精度
        # yolov8x.pt: ~136MB, 最高精度
        
        print(f"模型已加载到: {device}")
        print(f"模型类别数: {len(self.model.names)}")
        
        # 预热
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)
        print("模型预热完成")
    
    def detect(self, image, visualize=True):
        """
        执行检测
        
        Args:
            image: 输入图像(BGR格式)
            visualize: 是否返回可视化结果
            
        Returns:
            results: ultralytics Results对象
            vis_image: 可视化图像(如果visualize=True)
        """
        # 执行推理
        results = self.model(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )[0]
        
        if visualize:
            # 使用ultralytics内置的可视化
            vis_image = results.plot(
                line_width=2,
                font_size=12,
                labels=True,
                boxes=True,
                conf=True
            )
            return results, vis_image
        
        return results, None
    
    def get_detections(self, results):
        """
        从results对象中提取检测结果
        
        Returns:
            boxes: [N, 4] (x1, y1, x2, y2)
            scores: [N]
            class_ids: [N]
            class_names: [N] (类别名称列表)
        """
        boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        class_names = [results.names[i] for i in class_ids]
        
        return boxes, scores, class_ids, class_names
    
    def export_tensorrt(self, img_size=640, half=True):
        """
        导出TensorRT引擎
        
        Args:
            img_size: 输入图像尺寸
            half: 是否使用FP16精度
            
        Returns:
            engine_path: 生成的引擎文件路径
        """
        print(f"导出TensorRT引擎 (FP{'16' if half else '32'})...")
        
        # ultralytics会自动处理导出
        engine_path = self.model.export(
            format='engine',  # TensorRT
            imgsz=img_size,
            half=half,  # FP16
            simplify=True,  # 简化ONNX
            workspace=4,  # GPU内存工作空间(GB)
            verbose=True
        )
        
        print(f"TensorRT引擎已生成: {engine_path}")
        return engine_path
    
    def export_onnx(self, img_size=640):
        """
        导出ONNX模型
        
        Args:
            img_size: 输入图像尺寸
            
        Returns:
            onnx_path: 生成的ONNX文件路径
        """
        print("导出ONNX模型...")
        
        onnx_path = self.model.export(
            format='onnx',
            imgsz=img_size,
            simplify=True,
            opset=12,  # ONNX opset版本
            verbose=True
        )
        
        print(f"ONNX模型已生成: {onnx_path}")
        return onnx_path

class FPSCounter:
    """FPS计数器"""
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = []
    
    def update(self):
        now = time.time()
        self.timestamps.append(now)
        
        # 保持窗口大小
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
    
    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff > 0:
            return (len(self.timestamps) - 1) / time_diff
        return 0.0

def main():
    parser = argparse.ArgumentParser(description='YOLO物体检测 (Ultralytics)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='模型路径 (yolov8n/s/m/l/x.pt 或 .engine)')
    parser.add_argument('--source', type=str, default='0',
                       help='输入源: 摄像头ID/视频文件/图片路径')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.4,
                       help='NMS IOU阈值')
    parser.add_argument('--device', type=str, default='cpu',
                       help='运行设备: cpu, cuda:0, cuda:1, ...')
    parser.add_argument('--img-size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--export-tensorrt', action='store_true',
                       help='导出TensorRT引擎后退出')
    parser.add_argument('--export-onnx', action='store_true',
                       help='导出ONNX模型后退出')
    parser.add_argument('--half', action='store_true',
                       help='使用FP16精度(TensorRT导出)')
    parser.add_argument('--save-vid', type=str, default=None,
                       help='保存检测结果视频')
    args = parser.parse_args()
    
    # 初始化检测器
    detector = YOLODetector(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device=args.device
    )
    
    # 仅导出模式
    if args.export_tensorrt:
        engine_path = detector.export_tensorrt(
            img_size=args.img_size,
            half=args.half
        )
        print(f"\n导出完成!")
        print(f"使用方法: python {__file__} --model {engine_path} --device cuda:0")
        return
    
    if args.export_onnx:
        onnx_path = detector.export_onnx(img_size=args.img_size)
        print(f"\n导出完成!")
        print(f"使用方法: python {__file__} --model {onnx_path}")
        return
    
    # 打开输入源
    try:
        source = int(args.source)  # 摄像头ID
    except ValueError:
        source = args.source  # 视频文件或图片
    
    # 判断是图片还是视频
    if isinstance(source, str) and Path(source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # 单张图片模式
        image = cv2.imread(source)
        if image is None:
            print(f"无法读取图片: {source}")
            return
        
        print("处理图片...")
        results, vis_image = detector.detect(image, visualize=True)
        
        # 打印检测结果
        boxes, scores, class_ids, class_names = detector.get_detections(results)
        print(f"\n检测到 {len(boxes)} 个物体:")
        for i, (box, score, class_name) in enumerate(zip(boxes, scores, class_names)):
            x1, y1, x2, y2 = box
            print(f"  {i+1}. {class_name}: {score:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        # 显示结果
        cv2.imshow('Detection', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        if args.save_vid:
            output_path = Path(args.save_vid).with_suffix('.jpg')
            cv2.imwrite(str(output_path), vis_image)
            print(f"结果已保存: {output_path}")
    
    else:
        # 视频/摄像头模式
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"无法打开输入源: {source}")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n输入源信息:")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps_in}")
        
        # 视频写入器
        writer = None
        if args.save_vid:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                args.save_vid,
                fourcc,
                fps_in if fps_in > 0 else 30,
                (width, height)
            )
            print(f"将保存到: {args.save_vid}")
        
        # FPS统计
        fps_counter = FPSCounter(window_size=30)
        
        print("\n按键说明:")
        print("  q - 退出")
        print("  p - 暂停/继续")
        print("  s - 保存当前帧")
        
        paused = False
        frame_count = 0
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("视频结束或无法读取帧")
                        break
                    
                    frame_count += 1
                    
                    # 检测
                    start_time = time.time()
                    results, vis_image = detector.detect(frame, visualize=True)
                    infer_time = (time.time() - start_time) * 1000
                    
                    # 更新FPS
                    fps_counter.update()
                    fps = fps_counter.get_fps()
                    
                    # 获取检测结果
                    boxes, scores, class_ids, class_names = detector.get_detections(results)
                    
                    # 添加性能信息
                    info_y = 30
                    cv2.putText(vis_image, f"FPS: {fps:.1f}", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(vis_image, f"Inference: {infer_time:.1f}ms", (10, info_y + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(vis_image, f"Objects: {len(boxes)}", (10, info_y + 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 显示
                    cv2.imshow('YOLO Detection', vis_image)
                    
                    # 保存视频
                    if writer is not None:
                        writer.write(vis_image)
                    
                    # 每100帧打印一次统计
                    if frame_count % 100 == 0:
                        print(f"已处理 {frame_count} 帧, FPS: {fps:.1f}, 推理: {infer_time:.1f}ms")
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("暂停" if paused else "继续")
                elif key == ord('s'):
                    filename = f"detection_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, vis_image)
                    print(f"已保存: {filename}")
        
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"\n总计处理 {frame_count} 帧")
            print(f"平均FPS: {fps_counter.get_fps():.2f}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
TensorRT INT8量化校准器
从图像目录读取校准数据
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from pathlib import Path
import glob

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8校准器
    使用图像数据统计激活值范围
    """
    
    def __init__(self, calib_images, cache_file, input_shape=(640, 640), batch_size=1):
        """
        Args:
            calib_images: 校准图像目录或文件列表
            cache_file: 校准缓存文件
            input_shape: 输入尺寸(H, W)
            batch_size: 批大小
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_shape = input_shape
        
        # 加载图像列表
        if isinstance(calib_images, str):
            if Path(calib_images).is_dir():
                self.images = glob.glob(str(Path(calib_images) / '*.jpg')) + \
                             glob.glob(str(Path(calib_images) / '*.png'))
            else:
                self.images = [calib_images]
        else:
            self.images = calib_images
        
        if len(self.images) < 100:
            print(f"警告: 校准图像较少({len(self.images)}张),建议至少100张")
        
        print(f"校准数据集: {len(self.images)} 张图像")
        
        # 预处理所有图像
        self.data = self._load_images()
        self.batch_idx = 0
        
        # 分配GPU内存
        self.device_input = cuda.mem_alloc(
            self.batch_size * 3 * input_shape[0] * input_shape[1] * np.float32().itemsize
        )
    
    def _load_images(self):
        """预处理所有校准图像"""
        print("预处理校准图像...")
        data = []
        
        for img_path in self.images:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # 预处理(与推理时相同)
            img = cv2.resize(img, self.input_shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW
            
            data.append(img)
        
        data = np.array(data)
        print(f"✓ 已加载 {len(data)} 张校准图像")
        
        return data
    
    def get_batch_size(self):
        """返回批大小"""
        return self.batch_size
    
    def get_batch(self, names):
        """获取下一批数据"""
        if self.batch_idx < len(self.data):
            # 获取batch
            batch = self.data[self.batch_idx:self.batch_idx + self.batch_size]
            
            # 如果不够一个batch,用零填充
            if len(batch) < self.batch_size:
                padding = np.zeros(
                    (self.batch_size - len(batch), 3, self.input_shape[0], self.input_shape[1]),
                    dtype=np.float32
                )
                batch = np.concatenate([batch, padding], axis=0)
            
            # 拷贝到GPU
            cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
            
            self.batch_idx += self.batch_size
            return [int(self.device_input)]
        else:
            return None
    
    def read_calibration_cache(self):
        """读取校准缓存"""
        if Path(self.cache_file).exists():
            print(f"使用缓存的校准数据: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """写入校准缓存"""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"✓ 校准缓存已保存: {self.cache_file}")

# 使用示例
if __name__ == '__main__':
    # 创建校准器
    calibrator = ImageCalibrator(
        calib_images='path/to/calibration/images',
        cache_file='calibration.cache',
        input_shape=(640, 640)
    )
    
    print("校准器已创建")

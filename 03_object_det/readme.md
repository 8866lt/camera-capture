# TensorRT优化工具

将YOLO模型优化为TensorRT引擎,实现3-5倍加速。

## 快速开始

### 1. 导出ONNX
```bash
python export_onnx.py --weights yolov8n.pt --img-size 640
```

### 2. 构建TensorRT引擎
```bash
# FP32
python build_engine.py --onnx yolov8n.onnx

# FP16 (推荐)
python build_engine.py --onnx yolov8n.onnx --fp16

# INT8 (需要校准)
python build_engine.py \
    --onnx yolov8n.onnx \
    --int8 \
    --calib-images /path/to/calibration/images
```

### 3. 性能测试
```bash
python benchmark.py --engines yolov8n_fp16.engine
```

### 4. 使用引擎推理
```bash
cd ..
python yolo_detection.py --model models/yolov8n_fp16.engine --source 0
```

## 工具说明

| 工具 | 功能 |
|------|------|
| `export_onnx.py` | PyTorch → ONNX |
| `build_engine.py` | ONNX → TensorRT |
| `calibrator.py` | INT8量化校准 |
| `benchmark.py` | 性能测试 |
| `batch_convert.sh` | 批量转换 |

## 性能参考

Jetson Xavier NX, YOLOv8n, 640×640:

| 精度 | 延迟 | FPS | mAP@0.5 |
|------|------|-----|---------|
| PyTorch FP32 | 85ms | 12 | 37.3% |
| TensorRT FP32 | 42ms | 24 | 37.3% |
| TensorRT FP16 | 28ms | 36 | 37.2% |
| TensorRT INT8 | 18ms | 55 | 35.8% |

## 常见问题

### Q: TensorRT版本不兼容

引擎与TensorRT版本绑定,需要在目标设备上重新构建。

### Q: INT8精度下降太多

增加校准数据量,确保覆盖各种场景(室内/室外,白天/夜晚)。

### Q: 构建引擎很慢

正常现象,首次构建需要5-10分钟。后续使用缓存会很快。

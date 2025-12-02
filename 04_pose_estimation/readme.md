基于MediaPipe的人体姿态估计完整解决方案,适用于人形机器人视觉感知。

## 📦 快速开始

### 安装依赖

```bash
pip install mediapipe opencv-python numpy
```

### 基础使用

```bash
# 实时姿态检测
python mediapipe_pose.py --source 0

# 跌倒检测
python fall_detection.py --source 0

# 手势识别
python gesture_recognition.py --source 0

# 深蹲评估
python squat_evaluator.py --source 0
```

---

## 📂 模块说明

| 文件 | 功能 | 适用场景 |
|------|------|---------|
| `mediapipe_pose.py` | 基础姿态检测 | 通用姿态估计 |
| `fall_detection.py` | 跌倒检测 | 安防监控、老人看护 |
| `gesture_recognition.py` | 手势识别 | 人机交互、体感控制 |
| `squat_evaluator.py` | 运动姿态评估 | 健身应用、动作纠正 |

---

## 🎯 核心功能

### 1. 基础姿态检测

**输出:** 33个人体关键点(x, y, z, visibility)

**关键点索引:**
```
0: 鼻子
11-12: 肩膀
13-14: 肘
15-16: 腕
23-24: 髋部
25-26: 膝盖
27-28: 踝
```

**示例:**
```python
from mediapipe_pose import PoseDetector

detector = PoseDetector(model_complexity=1)
results, annotated = detector.detect(frame)

if results.pose_landmarks:
    landmarks = detector.get_all_landmarks(results)
    print(f"检测到 {len(landmarks)} 个关键点")
```

---

### 2. 跌倒检测

**判定依据:**
- 躯干角度 > 60° (接近水平)
- 髋部高度 > 0.8 (接近地面)
- 连续N帧满足条件

**参数调整:**
```python
detector = FallDetector(
    angle_threshold=60,      # 躯干角度阈值
    hip_threshold=0.8,       # 髋部高度阈值
    confidence_window=15,    # 时间窗口(帧)
    confidence_ratio=0.7     # 置信度比例
)
```

**典型输出:**
```
✓ 正常: angle=15°, hip=0.45
✗ 跌倒: angle=75°, hip=0.85
```

---

### 3. 手势识别

**支持的手势:**
- 挥手 (wave)
- 举手 (raise_hand)
- 双手举起 (hands_up)
- 指向左侧/右侧 (point_left/right)
- 双臂交叉 (arms_crossed)

**自定义手势:**
```python
# 在gesture_recognition.py中添加规则
def _match_gesture(self, features, landmarks):
    # 示例:检测"OK"手势
    if thumb_and_index_form_circle():
        return 'ok', 0.9
```

---

### 4. 运动姿态评估

**评估指标:**
- ✅ 膝盖角度 (70-100°)
- ✅ 背部角度 (< 20°)
- ✅ 膝盖不超脚尖
- ✅ 膝盖不内扣

**评分系统:**
```
100分: 完美姿态
90+分: 优秀
70+分: 良好
<70分: 需改进
```

**实时计数:**
- 自动识别深蹲的"起-蹲-起"循环
- 只在完整动作后计数

---

## ⚙️ 参数配置

### 模型复杂度

```python
model_complexity=0  # Lite:  快速但精度较低
model_complexity=1  # Full:  平衡模式(推荐)
model_complexity=2  # Heavy: 高精度但较慢
```

**性能对比(640×480):**

| 模型 | Jetson Nano | Jetson Xavier NX | 桌面CPU |
|------|-------------|------------------|---------|
| Lite | 25ms (40 FPS) | 12ms (83 FPS) | 15ms (66 FPS) |
| Full | 40ms (25 FPS) | 20ms (50 FPS) | 25ms (40 FPS) |
| Heavy | 70ms (14 FPS) | 38ms (26 FPS) | 45ms (22 FPS) |

### 置信度阈值

```python
min_detection_confidence=0.5  # 检测阈值(首次检测)
min_tracking_confidence=0.5   # 跟踪阈值(后续帧)
```

**建议设置:**
- 光线好、姿态清晰: 0.7
- 一般情况: 0.5 (默认)
- 遮挡较多、光线差: 0.3

---

## 🚀 性能优化

### 1. 降低分辨率

```python
# 从1080p降到480p
frame = cv2.resize(frame, (640, 480))
results, annotated = detector.detect(frame)

# 速度提升: 2-3x
# 精度损失: <5%
```

### 2. 跳帧处理

```python
frame_count = 0
process_every = 2  # 每2帧处理一次

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % process_every == 0:
        results, annotated = detector.detect(frame)
    
    # 使用上次结果显示
    cv2.imshow('Pose', annotated)
```

### 3. 多线程(高级)

```python
from threading import Thread
from queue import Queue

# 分离捕获和处理
capture_thread = Thread(target=capture_frames)
process_thread = Thread(target=process_poses)

capture_thread.start()
process_thread.start()
```

---

## 🔧 常见问题

### Q1: 检测不到人或关键点抖动

**解决:**
```python
# 1. 提高置信度
detector = PoseDetector(min_detection_confidence=0.7)

# 2. 改善光线
# - 确保正面光照
# - 避免逆光

# 3. 时间平滑
# - 使用移动平均滤波关键点
```

### Q2: Jetson上运行慢

**优化策略:**
```python
# 1. 使用Lite模型
model_complexity=0

# 2. 降低分辨率
frame = cv2.resize(frame, (320, 240))

# 3. 跳帧
process_every_n_frames = 2
```

### Q3: 侧面/背面检测失败

**原因:** MediaPipe主要针对正面/斜侧面训练

**解决:**
```python
# 检查可见性
def check_visibility(landmarks):
    left_visible = landmarks[11].visibility > 0.5  # 左肩
    right_visible = landmarks[12].visibility > 0.5  # 右肩
    
    if not (left_visible or right_visible):
        return "背面或被遮挡"
```

### Q4: 多人场景只检测一个

**解决:** 先用YOLO检测所有人,再逐个姿态估计

```python
# 伪代码
people_boxes = yolo.detect(frame)

for box in people_boxes:
    person_roi = crop(frame, box)
    pose_result = mediapipe_pose.process(person_roi)
```

---

## 📊 实战案例

### 案例1: 健身计数器

```python
class FitnessCounter:
    def __init__(self, exercise_type='squat'):
        self.count = 0
        self.state = 'up'
    
    def update(self, knee_angle):
        if self.state == 'up' and knee_angle < 100:
            self.state = 'down'
        elif self.state == 'down' and knee_angle > 140:
            self.state = 'up'
            self.count += 1
        
        return self.count
```

### 案例2: 虚拟试衣

```python
# 基于肩宽和身高估算衣服尺寸
shoulder_width = distance(left_shoulder, right_shoulder)
body_height = distance(nose, ankle)

if shoulder_width < 0.2 and body_height < 0.7:
    size = 'S'
elif shoulder_width < 0.25 and body_height < 0.8:
    size = 'M'
else:
    size = 'L'
```

### 案例3: 姿态矫正提醒

```python
if back_angle > 30:
    alert("请挺直背部!")

if knee_over_toe:
    alert("膝盖不要超过脚尖!")
```

---

## 📈 33个关键点详细说明

```
脸部 (0-10):
  0: nose (鼻子)
  1-2: left/right eye inner (内眼角)
  3-4: left/right eye (眼睛)
  5-6: left/right eye outer (外眼角)
  7-8: left/right ear (耳朵)
  9-10: mouth left/right (嘴角)

上肢 (11-22):
  11-12: left/right shoulder (肩膀)
  13-14: left/right elbow (肘)
  15-16: left/right wrist (腕)
  17-18: left/right pinky (小指)
  19-20: left/right index (食指)
  21-22: left/right thumb (拇指)

下肢 (23-32):
  23-24: left/right hip (髋)
  25-26: left/right knee (膝)
  27-28: left/right ankle (踝)
  29-30: left/right heel (脚跟)
  31-32: left/right foot index (脚尖)
```

---

## 🎓 进阶学习

### 3D坐标系统

```python
# z坐标是相对深度(相对于髋部中心)
landmark = results.pose_landmarks.landmark[15]  # 右手腕

x = landmark.x  # 归一化坐标 0-1
y = landmark.y  # 归一化坐标 0-1
z = landmark.z  # 相对深度(米)
visibility = landmark.visibility  # 可见性 0-1

# z < 0: 在身体前方
# z > 0: 在身体后方
```

### 动作识别(时序)

```python
# 收集30帧作为一个动作序列
sequence = []
for frame in video[:30]:
    landmarks = extract_landmarks(frame)
    sequence.append(landmarks)

# 输入LSTM模型分类
action = lstm_model.predict(sequence)
# 输出: "walking", "running", "jumping", etc.
```

---

## 📝 命令行参数

### mediapipe_pose.py

```bash
python mediapipe_pose.py \
    --source 0                # 输入源(摄像头/视频)
    --model 1                 # 模型复杂度(0/1/2)
    --confidence 0.5          # 检测置信度
    --output output.mp4       # 保存视频
```

### fall_detection.py

```bash
python fall_detection.py \
    --source 0                    # 输入源
    --angle-threshold 60          # 角度阈值
    --hip-threshold 0.8           # 髋部阈值
    --alarm-sound alarm.wav       # 报警声音
```

### gesture_recognition.py

```bash
python gesture_recognition.py \
    --source 0                # 输入源
```

### squat_evaluator.py

```bash
python squat_evaluator.py \
    --source 0                # 输入源
```

---

## 🔗 相关资源

**MediaPipe官方:**
- 文档: https://google.github.io/mediapipe/solutions/pose
- GitHub: https://github.com/google/mediapipe

**论文:**
- BlazePose: On-device Real-time Body Pose tracking (2020)

**配套文章:**
- 知乎: 《人形机器人视觉(四):MediaPipe人体姿态估计实战》

---

## 📄 License

MIT License

---

## 🤝 贡献

欢迎提交Issue和PR!

**开发计划:**
- [ ] 多人姿态检测
- [ ] 更多运动类型评估(俯卧撑、引体向上)
- [ ] 动作识别LSTM模型
- [ ] ROS2集成节点

---

## ✨ 快速测试

```bash
# 1. 安装依赖
pip install mediapipe opencv-python numpy

# 2. 运行基础检测
python mediapipe_pose.py --source 0

# 3. 尝试跌倒检测(假装摔倒)
python fall_detection.py --source 0

# 4. 尝试手势识别(挥手、举手)
python gesture_recognition.py --source 0

# 5. 做几个深蹲
python squat_evaluator.py --source 0
```

---

**最后更新:** 2025年12月

**版本:** v1.0.0


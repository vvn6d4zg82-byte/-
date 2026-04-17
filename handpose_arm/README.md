# HandPose X 机械臂控制

基于 handpose_x-main 手势识别模型的机械臂控制系统。

## 目录结构

```
handpose_arm/
├── main.py            # 主程序入口
├── hand_detector.py   # 手部检测模块（基于肤色分割）
├── gesture_utils.py   # 手势识别和关键点处理
├── arm_control.py     # 机械臂通信控制
├── visualizer.py      # 可视化工具
└── README.md
```

## 环境要求

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- PySerial

安装依赖：
```bash
pip install torch opencv-python numpy pyserial
```

## 模型文件

需要下载预训练模型：
- 下载链接: https://pan.baidu.com/s/1Ur6Ikp31XGEuA3hQjYzwIw
- 密码: 99f3
- 将模型文件放到: `handpose_x-main/weights/ReXNetV1-size-256-wingloss.pth`

## 使用方法

```bash
cd handpose_arm
python main.py
```

## 控制说明

- **手掌位置** → 控制机械臂基座和臂的角度
- **手势识别** → 自动控制夹爪开合
  - `fist` (握拳) → 夹爪闭合 (180°)
  - `five` (五指张开) → 夹爪张开 (90°)
- **稳定检测** → 手部位置稳定后自动发送控制指令
- **Q键** → 退出程序

## 舵机映射

| 舵机 | 功能 | 控制 |
|------|------|------|
| S1   | 基座旋转 | 水平位置映射 |
| S2   | 大臂 | 垂直位置映射 |
| S3   | 小臂 | 垂直位置映射（反向） |
| S4   | 夹爪 | 手势自动控制 |
| S5   | 旋转 | 水平位置映射（反向） |

## 串口配置

默认尝试连接: COM3-COM9
波特率: 115200

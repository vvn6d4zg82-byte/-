# TensorFlow.js 替代 MediaPipe 方案

## 概述

本项目提供了使用 **TensorFlow.js Hand Pose Detection** 替代 **MediaPipe** 的完整方案，适用于不想依赖 MediaPipe 的开发者。

## 新增文件

```
src/
├── hand_landmark/
│   ├── detector_tfjs.ts          # TensorFlow.js 手势检测器
│   ├── gesture_handler_tfjs.ts  # 手势处理器
│   └── VideoDetectorTFJS.vue    # Vue 组件
└── AppTFJS.vue                   # 入口页面
```

## 依赖对比

| 依赖 | 原方案 (MediaPipe) | 新方案 (TensorFlow.js) |
|------|-------------------|----------------------|
| 手势识别 | @mediapipe/tasks-vision | @tensorflow-models/hand-pose-detection |
| 运行时 | MediaPipe WASM | TensorFlow.js |
| WebAssembly | 必须 | 必须 |

## 安装

```bash
# 安装新的 npm 依赖
npm uninstall @mediapipe/tasks-vision
npm install @tensorflow-models/hand-pose-detection @tensorflow/tfjs

# 安装 Python 轻量依赖（移除 OpenCV）
pip install -r requirements_light.txt
```

## 启动后端

```bash
# 使用轻量版后端
python src-py/main_light.py

# 或使用 PyInstaller 打包
pyinstaller --noconfirm --distpath src-tauri/bin/ main_win.spec
```

## 手势支持

| 手势 | 动作 | 说明 |
|------|------|------|
| 食指竖起 | 鼠标移动 | 控制光标位置 |
| 食指+中指 | 左键点击 | 执行点击 |
| 食指+拇指捏合 | 滚动 | 上下滚动页面 |
| 四指竖起 | 快捷键 | 发送自定义快捷键 |
| 五指(手掌) | 暂停/恢复 | 长按切换识别状态 |
| 拇指+小指 | 语音识别 | 开始语音输入 |
| 拳头 | 停止语音 | 结束语音输入 |

## 机械臂控制

通过 WebSocket 接收前端手势数据，转换为串口指令：

```
WebSocket {"type":"mouse_move","data":{"x":100,"y":200}}
     ↓
Python 处理
     ↓
串口发送 "1{base_angle}\r\n"
     ↓
机械臂响应
```

## 配置说明

### 识别框设置

```typescript
app_store.config.boundary_left   // 识别区域左边界
app_store.config.boundary_top     // 识别区域上边界
app_store.config.boundary_width  // 识别区域宽度
app_store.config.boundary_height // 识别区域高度
```

### 串口配置

```python
# main_light.py
ser = serial.Serial('COM5', 115200, timeout=1)
```

修改为你的机械臂串口号和波特率。

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| / | GET | 健康检查 |
| /health | GET | 详细状态信息 |
| /shutdown | GET | 关闭服务 |
| /ws_lazyeat | WebSocket | 手势指令通道 |

## 性能对比

| 指标 | MediaPipe | TensorFlow.js |
|------|----------|--------------|
| 模型大小 | ~30MB | ~2MB |
| FPS | 30+ | 25-30 |
| 精度 | 较高 | 中等 |
| 兼容性 | 较好 | 更好 |
| 离线支持 | 是 | 是 |

## 故障排除

### 摄像头无法访问
1. 检查浏览器权限
2. 确保没有其他程序占用摄像头
3. 尝试刷新页面

### TensorFlow.js 加载失败
1. 确保网络连接正常（首次加载模型）
2. 检查浏览器是否支持 WebGL
3. 尝试使用 Chrome 或 Edge 浏览器

### 串口连接失败
1. 检查 COM 口是否正确
2. 确认串口驱动已安装
3. 检查机械臂波特率设置

## 扩展建议

1. **自定义手势**: 修改 `detector_tfjs.ts` 中的手势映射表
2. **多机械臂**: 在 `main_light.py` 中添加多个串口连接
3. **其他控制**: 在 Python 后端添加 GPIO、网络控制等

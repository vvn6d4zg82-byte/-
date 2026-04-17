#!/usr/bin/env python3
"""
Lazyeat 轻量版后端服务
不使用 OpenCV 和 MediaPipe，仅通过 WebSocket 接收手势指令
支持鼠标键盘控制和串口机械臂通信
"""

import json
import os
import sys
import signal
import time
from typing import Dict, List, Optional, Union

import serial
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Button, Controller

try:
    ser = serial.Serial('COM5', 115200, timeout=1)
    time.sleep(2)
    print("串口已连接")
except:
    ser = None
    print("串口未连接")


def move(s: int, a: int) -> None:
    """发送串口指令到机械臂"""
    if ser:
        try:
            ser.write(f"{s}{a}\r\n".encode())
        except:
            pass


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connection: Optional[WebSocket] = None


class WebSocketMessage:
    """WebSocket消息数据类"""
    type: str
    msg: str = ""
    title: str = "提示"
    duration: int = 1
    data: Dict = None

    def __init__(self, **kwargs):
        self.type = kwargs.get('type', '')
        self.msg = kwargs.get('msg', '')
        self.title = kwargs.get('title', '提示')
        self.duration = kwargs.get('duration', 1)
        self.data = kwargs.get('data', {"x": 0, "y": 0, "key_str": ""})


class WebSocketMessageType:
    """WebSocket消息类型常量"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_SCROLL_UP = "mouse_scroll_up"
    MOUSE_SCROLL_DOWN = "mouse_scroll_down"
    SEND_KEYS = "send_keys"
    VOICE_RECORD = "voice_record"
    VOICE_STOP = "voice_stop"


class GestureHandler:
    """手势控制器 - 执行鼠标键盘操作和机械臂控制"""
    
    def __init__(self):
        self.keyboard = KeyboardController()
        self.mouse = Controller()
        self.last_move_time = 0

    def move_mouse(self, x: int, y: int) -> None:
        """移动鼠标到指定位置"""
        self.mouse.position = (x, y)
        
        now = time.time()
        if now - self.last_move_time > 0.08:
            w, h = 1920, 1080
            new_base = int(15 + 150 * x / w)
            new_arm = int(15 + 150 * y / h)
            new_arm2 = int(165 - 150 * y / h)
            new_rot = int(180 - 180 * x / w)
            move(1, new_base)
            move(2, new_arm)
            move(3, new_arm2)
            move(5, new_rot)
            self.last_move_time = now

    def click_mouse(self) -> None:
        """点击鼠标左键"""
        self.mouse.click(Button.left)
        move(4, 180)

    def scroll_up(self) -> None:
        """向上滚动"""
        self.mouse.scroll(0, 1)
        move(3, 30)

    def scroll_down(self) -> None:
        """向下滚动"""
        self.mouse.scroll(0, -1)
        move(3, 150)

    def send_keys(self, key_str: str) -> None:
        """发送按键事件（支持组合键）"""
        try:
            keys = [self._parse_key(k) for k in key_str.split("+")]
            self._execute_keys(keys)
        except Exception as e:
            print(f"发送按键失败: {e}")

    def _parse_key(self, key_str: str) -> Union[str, Key]:
        """解析单个按键"""
        key_str = key_str.strip().lower()
        
        if hasattr(Key, key_str):
            return getattr(Key, key_str)
        elif len(key_str) == 1:
            return key_str
        elif key_str.startswith("f"):
            try:
                return getattr(Key, key_str)
            except AttributeError:
                raise ValueError(f"无效的功能键: {key_str}")
        else:
            raise ValueError(f"无效的按键: {key_str}")

    def _execute_keys(self, keys: List[Union[str, Key]]) -> None:
        """执行按键序列"""
        pressed_keys = []
        try:
            for key in keys:
                self.keyboard.press(key)
                pressed_keys.append(key)
        finally:
            for key in reversed(pressed_keys):
                self.keyboard.release(key)


class VoiceHandler:
    """语音处理控制器（可选功能）"""
    
    def __init__(self):
        self.controller = None
        self.is_recording = False

    async def start_recording(self, websocket: WebSocket) -> None:
        """开始录音"""
        if self.controller and not self.is_recording:
            self.controller.start_record_thread()
            self.is_recording = True

    async def stop_recording(self, websocket: WebSocket, gesture_handler: GestureHandler) -> None:
        """停止录音并处理结果"""
        if self.controller and self.is_recording:
            self.controller.stop_record()
            self.is_recording = False
            
            text = self.controller.transcribe_audio()
            if text:
                try:
                    gesture_handler.keyboard.type(text)
                    gesture_handler.keyboard.tap(Key.enter)
                except Exception as e:
                    print(f"语音输入失败: {e}")


@app.get("/")
def read_root():
    return "Lazyeat Backend Ready"


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "serial_connected": ser is not None,
        "timestamp": time.time()
    }


@app.get("/shutdown")
def shutdown():
    os.kill(os.getpid(), signal.SIGINT)


@app.websocket("/ws_lazyeat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点 - 接收前端手势指令"""
    global active_connection
    
    await websocket.accept()
    active_connection = websocket
    
    gesture_handler = GestureHandler()
    voice_handler = VoiceHandler()
    
    print("客户端已连接")
    
    try:
        while True:
            data_str = await websocket.receive_text()
            await _handle_message(data_str, websocket, gesture_handler, voice_handler)
    except WebSocketDisconnect:
        print("客户端已断开")
        active_connection = None
    except Exception as e:
        print(f"WebSocket处理错误: {e}")
        active_connection = None


async def _handle_message(
    data_str: str,
    websocket: WebSocket,
    gesture_handler: GestureHandler,
    voice_handler: VoiceHandler
) -> None:
    """处理WebSocket消息"""
    try:
        message = WebSocketMessage(**json.loads(data_str))
        data = message.data

        if message.type == WebSocketMessageType.MOUSE_MOVE:
            gesture_handler.move_mouse(data.get("x", 0), data.get("y", 0))
        elif message.type == WebSocketMessageType.MOUSE_CLICK:
            gesture_handler.click_mouse()
        elif message.type == WebSocketMessageType.MOUSE_SCROLL_UP:
            gesture_handler.scroll_up()
        elif message.type == WebSocketMessageType.MOUSE_SCROLL_DOWN:
            gesture_handler.scroll_down()
        elif message.type == WebSocketMessageType.SEND_KEYS:
            gesture_handler.send_keys(data.get("key_str", ""))
        elif message.type == WebSocketMessageType.VOICE_RECORD:
            await voice_handler.start_recording(websocket)
        elif message.type == WebSocketMessageType.VOICE_STOP:
            await voice_handler.stop_recording(websocket, gesture_handler)

    except json.JSONDecodeError:
        print("无效的JSON数据")
    except Exception as e:
        print(f"处理消息失败: {e}")


if __name__ == '__main__':
    port = 62334
    
    print(f"Starting Lazyeat Backend at http://127.0.0.1:{port}")
    print(f"Serial port: {'COM5' if ser else 'Not connected'}")
    print(f"WebSocket endpoint: ws://127.0.0.1:{port}/ws_lazyeat")
    
    uvicorn.run(app, host="127.0.0.1", port=port)

import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("1. 导入模块...")

try:
    from PyQt5.QtWidgets import QApplication
    print("   PyQt5 OK")
except Exception as e:
    print(f"   错误: {e}")
    sys.exit(1)

try:
    sys.path.insert(0, r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main')
    from hand_detector import detect_hand_bbox
    print("   hand_detector OK")
except Exception as e:
    print(f"   错误: {e}")
    detect_hand_bbox = None

try:
    from gesture_utils import get_palm_center, recognize_gesture
    print("   gesture_utils OK")
except Exception as e:
    print(f"   错误: {e}")
    get_palm_center = None

try:
    from arm_control import ArmController
    print("   arm_control OK")
except Exception as e:
    print(f"   错误: {e}")
    ArmController = None

print("2. 启动GUI...")

try:
    from ai_gui import AIGUI
    
    app = QApplication(sys.argv)
    window = AIGUI()
    
    print("   GUI创建成功")
    print("3. 关闭窗口测试...")
    
    import threading
    timer = threading.Timer(3, lambda: print("   3秒超时，自动退出"))
    timer.start()
    
    window.show()
    app.quit()
    
    print("   测试完成")
except Exception as e:
    import traceback
    print(f"   错误: {e}")
    traceback.print_exc()
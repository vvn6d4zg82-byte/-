import serial
import time

class ArmController:
    def __init__(self, ports=None, baudrate=115200, timeout=1):
        self.ports = ports or ['COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'COM3', 'COM4']
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.connected = False
        
    def connect(self):
        for port in self.ports:
            try:
                self.ser = serial.Serial(port, self.baudrate, timeout=self.timeout)
                time.sleep(2)
                self.connected = True
                print(f"Connected to robotic arm on {port}")
                return True
            except:
                continue
        
        print("Warning: Could not connect to any serial port")
        self.connected = False
        return False
    
    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected from robotic arm")
        self.connected = False
    
    def send_command(self, servo_id, angle):
        if not self.connected or not self.ser:
            return False
        
        try:
            cmd = f"{servo_id}{angle}\r\n"
            self.ser.write(cmd.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def move_servo(self, servo_id, angle):
        return self.send_command(servo_id, angle)
    
    def set_all_servos(self, base, arm1, arm2, gripper, rotation):
        self.move_servo(1, base)
        time.sleep(0.05)
        self.move_servo(2, arm1)
        time.sleep(0.05)
        self.move_servo(3, arm2)
        time.sleep(0.05)
        self.move_servo(4, gripper)
        time.sleep(0.05)
        self.move_servo(5, rotation)
    
    def is_connected(self):
        return self.connected and self.ser and self.ser.is_open

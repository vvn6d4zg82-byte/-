"""
AI Controller Package
- data_collector: 收集手势-舵机映射数据
- model_trainer: 训练映射模型
- ai_controller: 使用AI模型控制机械臂
"""

from .data_collector import DataCollector, ManualDataCollector
from .model_trainer import ModelTrainer
from .ai_controller import AIController, AIRobotController

__all__ = [
    'DataCollector',
    'ManualDataCollector', 
    'ModelTrainer',
    'AIController',
    'AIRobotController'
]
import logging

def setup_logging(log_file="app.log", level=logging.INFO):
    """
    设置日志记录配置。
    
    :param log_file: 日志文件路径
    :param level: 日志级别
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path, test_size=0.2, random_state=42):
    """
    加载数据并分割为训练集和测试集。
    
    :param file_path: 数据文件路径
    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 训练集和测试集
    """
    data = pd.read_csv(file_path)
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data


import json

class ConfigManager:
    def __init__(self, config_file):
        """
        初始化配置管理器。
        
        :param config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        """
        加载配置文件。
        
        :return: 配置字典
        """
        with open(self.config_file, 'r') as f:
            return json.load(f)

    def get(self, key, default=None):
        """
        获取配置项。
        
        :param key: 配置项键
        :param default: 默认值
        :return: 配置项值
        """
        return self.config.get(key, default)
    
    
# 定义一些常量
MODEL_NAME = "deepseek_distill_r1_1.5B"
DATA_DIR = "data/"
MODEL_DIR = "models/"
LOG_FILE = "app.log"

import os
import logging
from utils.logging import setup_logging
from utils.data_loader import load_data
from utils.config_manager import ConfigManager
from utils.constants import MODEL_NAME, DATA_DIR, MODEL_DIR, LOG_FILE
from src.inference import DeepSeekDistillInference

def run():
    # 设置日志记录
    setup_logging(log_file=LOG_FILE)
    logger = logging.getLogger(__name__)
    logger.info("Starting the inference process...")

    # 加载配置文件
    config_manager = ConfigManager("config/model_config.json")
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    data_file = os.path.join(DATA_DIR, "input_data.csv")

    # 加载数据
    train_data, test_data = load_data(data_file)
    logger.info(f"Loaded data with {len(train_data)} training samples and {len(test_data)} test samples.")

    # 初始化模型
    inference_tool = DeepSeekDistillInference(model_path)
    logger.info("Model loaded successfully.")

    # 输入提示
    prompt = "Once upon a time in a land far, far away,"
    max_length = config_manager.get("max_length", 100)
    temperature = config_manager.get("temperature", 1.0)

    # 生成文本
    generated_text = inference_tool.generate_text(prompt, max_length=max_length, temperature=temperature)
    logger.info("Generated text successfully.")
    print("Generated Text:")
    print(generated_text)

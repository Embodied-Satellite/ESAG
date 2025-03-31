import yaml
import os
import logging
# 定义配置文件路径
CONFIG_FILE = os.path.join("config/config.yaml")
logging.basicConfig(level=logging.INFO)

def load_config():
    """
    加载配置文件 config.yaml

    Returns:
        dict: 配置文件内容
    """
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {CONFIG_FILE}")
        logging.error(f"配置文件未找到: {CONFIG_FILE}")
    except yaml.YAMLError as e:
        raise ValueError(f"解析配置文件失败: {e}")
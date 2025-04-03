import yaml
import os
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 定义配置文件路径
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config/config.yaml")
logging.basicConfig(level=logging.INFO)

def load_config():
    """
    加载配置文件 config.yaml，并将相对路径转换为绝对路径。

    Returns:
        dict: 配置文件内容
    """
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # 动态更新文件路径为绝对路径
        file_paths = config.get("file_paths", {})
        for key, relative_path in file_paths.items():
            file_paths[key] = os.path.join(PROJECT_ROOT, relative_path)

        config["file_paths"] = file_paths
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {CONFIG_FILE}")
    except yaml.YAMLError as e:
        raise ValueError(f"解析配置文件失败: {e}")
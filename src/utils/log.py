import logging
import os
from datetime import datetime
from src.utils.config import load_config

# 加载配置
config = load_config()
log_config = config["logging"]

# 定义日志文件路径
LOG_DIR = log_config["log_dir"]
LOG_FILE = os.path.join(LOG_DIR, f"{log_config['log_file_prefix']}_{datetime.now().strftime('%Y-%m-%d')}.log")

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志记录
logging.basicConfig(
    filename=LOG_FILE,
    level=getattr(logging, log_config["log_level"].upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 获取日志记录器
def get_logger(name):
    """
    获取一个日志记录器实例。

    Args:
        name (str): 日志记录器的名称。

    Returns:
        logging.Logger: 配置好的日志记录器。
    """
    return logging.getLogger(name)
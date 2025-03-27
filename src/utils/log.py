import logging
import os
from datetime import datetime

# 定义日志文件路径
LOG_DIR = "../../logs"
LOG_FILE = os.path.join(LOG_DIR, f"run_agent_{datetime.now().strftime('%Y-%m-%d')}.log")

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志记录
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
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
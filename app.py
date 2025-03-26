import sys
import logging
from src.agents.task_planning_agent import master_agent
from src.utils.log import get_logger

# 配置日志
logger = get_logger("App")

def main():
    """
    主函数，用于运行 master_agent 并处理用户输入。
    """
    try:
        # 示例用户输入
        user_input = "监测杭州西湖交通情况，给出卫星调度建议"
        
        # 调用 master_agent 处理任务
        logger.info(f"开始处理用户输入: {user_input}")
        response = master_agent.print_response(user_input, stream=True)
        
        # 记录日志
        logger.info("任务调度完成，结果: %s", response)
    except Exception as e:
        logger.error("任务调度失败: %s", str(e))
        print(f"任务调度失败: {e}")

if __name__ == "__main__":
    main()
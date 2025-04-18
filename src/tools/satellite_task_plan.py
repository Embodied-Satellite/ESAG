import os
import json
from typing import List

from agno.agent import Agent
from agno.tools import Toolkit
from agno.utils.log import logger

from src.Tre_starlink.satellite import satellite_plan_tool

class SatellitePlanTool(Toolkit):
    """卫星任务规划工具"""
    def __init__(self):
        super().__init__(name="Satellite Plan Tool")
        self.register(self.run_plan_tool)

    def get_satellite_plan(self, task_id: str, observation_time: list) -> str:
        """
        读取 task_execute.json 重新生成任务规划。

        Args:
            task_id (str): 任务ID。
            observation_time (list): 观测时间列表。

        Returns:
            str: 任务规划结果的 JSON 字符串。
        """

        task_execute_path = os.path.join(os.getcwd(), "src/Tre_starlink/dataset/task_execute.json")

        # 检查文件是否存在
        if not os.path.exists(task_execute_path):
            raise FileNotFoundError(f"文件 {task_execute_path} 不存在")
        
        try:
            with open(task_execute_path, "r", encoding="utf-8") as file:
                task_execute_data = json.load(file)
        except json.JSONDecodeError:
            raise ValueError(f"文件 {task_execute_path} 不是有效的 JSON 格式")
        
        # 获取任务和卫星当前状态
        tasks = task_execute_data.get("task", {})
        satellite_cur = task_execute_data.get("satellite_cur", "")
        if not tasks or not satellite_cur:
            raise ValueError("task_execute.json 文件内容不完整，缺少 'task' 或 'satellite_cur' 字段")

        # 解析任务并生成规划结果
        task_plan_results = []
        for satellite_id, task_list in tasks.items():
            if len(task_list) < 2:
                continue  # 跳过任务不足两行的情况

            # 提取任务信息
            start_task = task_list[0]
            end_task = task_list[1]

            # 解析开始任务
            start_time = start_task.split(" ")[0].split(":")[1]
            slew_angle = float(start_task.split("P:")[1])

            # 解析结束任务
            end_time = end_task.split(" ")[0].split(":")[1]

            # 构建任务规划结果
            task_plan_result = {
                "task_id": task_id,
                "satellite_id": satellite_id,
                "observation_time": observation_time,
                "slew_angle": slew_angle,
                "solar_panel_angle": '45',  # 如果适用，可以从其他字段中提取
            }
            task_plan_results.append(task_plan_result)

        return json.dumps(task_plan_results, ensure_ascii=False, indent=4)

    
    def run_plan_tool(self, params: dict) -> str:
        """
        任务规划工具主入口，返回JSON字符串
        必须返回字符串类型以符合AGNO框架要求
        
        Args:
            params (dict): 任务参数，包含任务ID、任务类型、任务优先级、时间优先级、质量优先级、有效期限、区域大小、云层覆盖率等。
        Returns:
            str: 任务规划结果的 JSON 字符串。
        """

        required = ["task_id", "observation_time"]
        if not all(k in params for k in required):
            return "Missing required parameters"
        
        task_id = params["task_id"]
        observation_time = params["observation_time"]
        # location = params["location"]

        satellite_plan = self.get_satellite_plan(task_id, observation_time)
        
        if not satellite_plan:
            return "Satellite plan not found."
                
        return satellite_plan

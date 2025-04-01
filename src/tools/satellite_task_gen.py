import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from agno.tools import Toolkit
from agno.utils.log import logger
from src.tools.commparse import run_gen_tool

class SatelliteGenTool(Toolkit):
    def __init__(self):
        super().__init__(name="Satellite Task Generation Tool")
        self.register(self.run_tool)

    def get_satellite_task_tool(self, user_input: str):
        # 调用 satellite_plan_tool 函数，获取卫星计划结果
        satellite_gen_result = run_gen_tool(user_input)
        # 返回获取到的卫星计划结果
        return satellite_gen_result


    def run_tool(self, query: str):
        """
        The tool for the Satellite Task Generation.
        
        Args:
            query (str): The query to be processed.
            
        Returns:
            satellite_task_list (dict): The satellite task list in JSON format.
        
        """
        try:
            task = self.get_satellite_task_tool(query)
            return task.to_json()
        
        except Exception as e:
            logger.error(f"任务解析失败: {e}")
            return {"error": "任务解析失败，请检查输入。"}
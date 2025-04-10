import os
import json
from typing import List

from agno.agent import Agent
from agno.tools import Toolkit
from agno.utils.log import logger

from src.Tre_starlink.satellite import satellite_plan_tool

class SatellitePlanTool(Toolkit):
    def __init__(self):
        super().__init__(name="Satellite Plan Tool")
        self.register(self.run_tool)

    def get_satellite_plan(self):
        # 调用 satellite_plan_tool 函数，获取卫星计划结果
        satellite_plan_result = satellite_plan_tool()
        # 返回获取到的卫星计划结果
        return satellite_plan_result
    
    def run_tool(self, query: str):
        """
        The tool reads a JSON to get the satellite plan for the mission.
        
        Args:
            query (str): The query to be processed.
            
        Returns:
            satellite_plan (dict): The satellite plan.
        
        """
        
        satellite_plan = self.get_satellite_plan()
        if not satellite_plan:
            return "Satellite plan not found."
                
        return satellite_plan
        


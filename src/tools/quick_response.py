import os
import json
from typing import List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.tools import Toolkit
from agno.utils.log import logger


class QuickResponse(Toolkit):
    def __init__(self):
        super().__init__(name="Quick Response Tool")
        self.register(self.run_tool)

    
    def run_tool(self, satellite_id: str,target_area: str, observation_type: str, onboard_resources: dict):
        """
        quick_response_tool Calculates the optimal satellite selection plan.
        
        Args:
            satellite_id (str) = The satellite ID.
            target_area (str): The target area for observation.
            observation_type (str): The type of observation.
            onboard_resources (dict): The parameters of onboard resources.
                        
        Returns:
            plan result (str): JSON content.
        
        """
        
        optimal_satellite = f"成功调用快速响应规划智能体： Optimal {satellite_id} selected for {target_area} with observation type {observation_type} using resources {onboard_resources}"
        
        # 返回成功调用代码
        logger.debug(f"Optimal satellite selection tool called successfully with target_area: {target_area}, observation_type: {observation_type}, resources: {onboard_resources} ")
        return optimal_satellite

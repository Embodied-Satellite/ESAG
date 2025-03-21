from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

class GetSateSchema(BaseModel):
    satellite_id: str = Field(description = "卫星ID")
    target_area: str = Field(description = "观测区域：如杭州、北京、上海等")
    observation_type: str = Field(description = "观测类型，如点目标抓取、区域目标捕捉、定点跟踪等")
    onboard_resources: dict = Field(description = "卫星资源参数，如卫星位置、能源、载荷、存储等")
    

@tool(args_schema = GetSateSchema)
def get_coverage_optimization_tool(target_area: str, observation_type: str, onboard_resources: dict, special_config_param: RunnableConfig):
    """
    coverage_optimization_tool Calculates the optimal satellite selection plan.

    Args: 
        satellite_id (str) = The satellite ID.
        target_area (str): The target area for observation.
        observation_type (str): The type of observation.
        onboard_resources (dict): The parameters of onboard resources.
    """
    # 进行计算获取最优卫星选择方案
    optimal_satellite = f"成功调用覆盖优化智能体 coverage_optimization_tool Optimal satellite selected for {target_area} with observation type {observation_type} using resources {onboard_resources} by team member {special_config_param['configurable']['user']}"
    
    # 返回成功调用代码
    logger.debug(f"Optimal satellite selection tool called successfully with target_area: {target_area}, observation_type: {observation_type}, resources: {onboard_resources}, user: {special_config_param['configurable']['user']}")
    return optimal_satellite
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from agno.tools import Toolkit
from agno.utils.log import logger


class Task:
    """观测任务类，结构化存储任务信息"""

    def __init__(self, task_id: int, location: str, task_type: str, latitude: float, longitude: float,
                 task_priority: int, time_priority: int, quality_priority: int,
                 validity_period: Tuple[str, str], area_size: Optional[float] = None,
                 weather: Optional[str] = None):
        self.task_id = task_id
        self.location = location
        self.task_type = task_type
        self.latitude = latitude
        self.longitude = longitude
        self.task_priority = task_priority
        self.time_priority = time_priority
        self.quality_priority = quality_priority
        self.validity_period = validity_period
        self.area_size = area_size
        self.weather = weather
        self.validate_task()


    def validate_task(self):
        """验证任务参数是否合理"""
        if self.task_type not in ["point_target", "area_target", "relay_tracking"]:
            raise ValueError(f"无效的任务类型: {self.task_type}")

        if not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError(f"无效的坐标: lat={self.latitude}, lon={self.longitude}")

        if self.task_type == "area_target" and (self.area_size is None or self.area_size <= 0):
            raise ValueError("区域任务必须提供有效的区域大小")

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            str(self.task_id): {
                "location": self.location,
                "LLA": [self.longitude, self.latitude, 0],
                "task_priority": self.task_priority,
                "time_priority": self.time_priority,
                "quality_priority": self.quality_priority,
                "cloudrate": self.weather if isinstance(self.weather, list) else [],
                "Validity_period": self.validity_period,
                "Observation_mode": "single" if self.task_type == "point_target" else "scan",
                "location_type": "area" if self.task_type == "area_target" else "point",
                "area_size": self.area_size if self.area_size else 0
            }
        }

    def to_json(self) -> str:
        """以 JSON 格式返回任务信息"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __repr__(self):
        return f"Task({self.to_json()})"


class SatelliteGenTool(Toolkit):
    """卫星任务生成工具"""
    def __init__(self):
        super().__init__(name="Satellite Task Generation Tool")
        self.register(self.run_tool)

        # 初始化 API Keys
        self.amap_key = os.getenv("AMAP_KEY", "4b80d2b289f03e3de147c8e82364eab6")
        self.caiyun_api_key = os.getenv("CAIYUN_API_KEY", "67JCvZ2qyIy9B8Om")

        # 重要位置的缓存
        self.important_locations = {
            "杭州西湖": [120.154974, 30.251381, 0],
            "之江实验室": [119.902168, 30.267847, 0],
            "杭州市区": [120.192267, 30.254933, 0]
        }
        self.register(self.run_tool)


    def parse_command(self, user_input: str) -> Task:
        """解析用户输入，返回 Task 对象"""
        task_params = self._parse_user_input(user_input)
        coordinates = self._get_geocode(task_params["location"])
        start_time, end_time = self._calculate_time_range(task_params["validity_period_days"])

        return Task(
            task_id=0,
            location=task_params["location"],
            task_type="area_target" if task_params["is_area"] else "point_target",
            latitude=coordinates["center"][1],
            longitude=coordinates["center"][0],
            task_priority=task_params["task_priority"],
            time_priority=task_params["time_priority"],
            quality_priority=task_params["quality_priority"],
            validity_period=(start_time, end_time),
            area_size=task_params.get("area_size"),
        )

    def _parse_user_input(self, user_input: str) -> Dict:
        """模拟 LLM 解析用户输入"""
        # 模拟解析结果
        return {
            "location": "杭州西湖",
            "is_area": False,
            "observation_mode": "single",
            "task_priority": 5,
            "time_priority": 5,
            "quality_priority": 5,
            "validity_period_days": 2,
            "area_size": None
        }

    def _get_geocode(self, location: str) -> Dict:
        """调用高德 API 获取经纬度"""
        if location in self.important_locations:
            return {"center": self.important_locations[location]}

        params = {"key": self.amap_key, "address": location, "output": "json"}
        try:
            response = requests.get("https://restapi.amap.com/v3/geocode/geo", params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "1" and data.get("count") != "0":
                lng, lat = map(float, data["geocodes"][0]["location"].split(","))
                return {"center": [lng, lat, 0]}
        except requests.RequestException as e:
            logger.error(f"获取地理编码失败: {e}")
        return {"center": [0, 0, 0]}

    def _calculate_time_range(self, validity_days: int) -> Tuple[str, str]:
        """计算任务的开始时间和结束时间"""
        now = datetime.now()
        end_time = now + timedelta(days=validity_days)
        return now.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')

    def run_tool(self, query: str):
        """

        The tool for the Satellite Task Generation.
        
        Args:
            query (str): The query to be processed.
            
        Returns:
            satellite_task_list (dict): The satellite task list in JSON format.
        
        """
        try:
            task = self.parse_command(query)
            return task.to_dict()
        except Exception as e:
            logger.error(f"任务解析失败: {e}")
            return {"error": "任务解析失败，请检查输入。"}
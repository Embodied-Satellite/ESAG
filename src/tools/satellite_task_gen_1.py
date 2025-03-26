import os
import json
import requests
import re
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

from agno.agent import Agent
from agno.tools import Toolkit
from agno.utils.log import logger

from src.Tre_starlink.satellite import satellite_plan_tool


AMAP_KEY = os.getenv("AMAP_KEY", "4b80d2b289f03e3de147c8e82364eab6")  
CAIYUN_API_KEY = os.getenv("CAIYUN_API_KEY", "67JCvZ2qyIy9B8Om")

class Task:
    """ 观测任务类，结构化存储任务信息 """

    def __init__(self, task_id: int, location: str, task_type: str, latitude: float, longitude: float, 
                 task_priority: int, time_priority: int, quality_priority: int,
                 validity_period: Tuple[str, str], area_size: Optional[float] = None, 
                 weather: Optional[str] = None):
        """
        初始化任务
        :param task_id: 任务编号
        :param location: 目标位置名称
        :param task_type: 任务类型（point_target, area_target, relay_tracking）
        :param latitude: 目标纬度
        :param longitude: 目标经度
        :param task_priority: 任务优先级（1-5）
        :param time_priority: 时间优先级（1-5）
        :param quality_priority: 质量优先级（1-5）
        :param validity_period: 任务有效期（开始时间, 结束时间）
        :param area_size: 面积目标时的大小（km²）
        :param weather: 观测目标区域的天气情况
        """
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
        """ 验证任务参数是否合理 """
        if self.task_type not in ["point_target", "area_target", "relay_tracking"]:
            raise ValueError(f"无效的任务类型: {self.task_type}")

        if not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError(f"无效的坐标: lat={self.latitude}, lon={self.longitude}")

        if self.task_type == "area_target" and (self.area_size is None or self.area_size <= 0):
            raise ValueError("区域任务必须提供有效的区域大小")

    def to_dict(self) -> Dict:
        """ 转换为字典格式，符合 output 结构 """
        return {
            str(self.task_id):{"location": self.location,
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
        """ 以 JSON 格式返回任务信息 """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __repr__(self):
        return f"Task({self.to_json()})"



class SatelliteGenTool(Toolkit):
    def __init__(self):
        super().__init__(name="Satellite Plan Generation Tool")
        """
        :param amap_key: 高德 API Key
        :param caiyun_api_key: 彩云 API Key
        """
        self.amap_key = AMAP_KEY
        self.caiyun_api_key = CAIYUN_API_KEY

        self.amap_geocode_url = "https://restapi.amap.com/v3/geocode/geo"
        self.weather_api_url = "https://api.caiyunapp.com/v2.6/{API_KEY}/{lon},{lat}/hourly"

        self.satellite_camera_width = 10.0

        self.important_locations = {
            "杭州西湖": [120.154974, 30.251381, 0],
            "之江实验室": [119.902168, 30.267847, 0],
            "杭州市区": [120.192267, 30.254933, 0]
        }
    
        self.register(self.run_tool)


    def _get_geocode(self, location: str) -> Dict:
        """调用高德 API 获取经纬度"""
        if location in self.important_locations:
            return {"center": self.important_locations[location]}

        params = {"key": self.amap_key, "address": location, "output": "json"}
        try:
            response = requests.get(self.amap_geocode_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "1" and data.get("count") != "0":
                lng, lat = map(float, data["geocodes"][0]["location"].split(","))
                return {"center": [lng, lat, 0]}
        except requests.RequestException as e:
            logger.error(f"获取地理编码失败: {e}")
        return {"center": [0, 0, 0]}

    def _fetch_weather(self, lon: float, lat: float) -> str:
        """调用彩云 API 获取天气"""
        url = self.weather_api_url.format(API_KEY=self.caiyun_api_key, lon=lon, lat=lat)
        try:
            response = requests.get(url, params={"hourlysteps": 48})
            response.raise_for_status()
            data = response.json()
            raw_cloudrate = data.get("result", {}).get("hourly", {}).get("cloudrate", [])
        
            # 格式化时间
            formatted_cloudrate = [
                {
                    "datetime": datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S"),
                    "value": entry["value"]
                }
                for entry in raw_cloudrate
            ]

            return {"cloudrate": formatted_cloudrate}
        except requests.RequestException as e:
            logger.error(f"获取天气失败: {e}")
        return {"cloudrate": []}

    
    def _calculate_time_range(self, validity_days: int) -> Tuple[str, str]:
        """计算任务的开始时间和结束时间"""
        now = datetime.now()
        end_time = now + timedelta(days=validity_days)
        return now.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')

    
    def run_tool(self, query: str):
        """
        Run the satellite plan generation tool with the given query.
        
        Args:
            query (str): The query string containing the task parameters.
            
            
        Returns:
            dict: A dictionary containing the task information.
        """
        if isinstance(query, dict):
            query = json.dumps(query)
        elif not query.strip().startswith("{"):
            # 如果 query 是逗号分隔的字符串，尝试解析
            try:
                keys = ["location", "task_type", "is_area", "task_priority", "time_priority", "quality_priority", "validity_period_days", "area_size"]
                values = query.split(",")
                query = json.dumps(dict(zip(keys, values)))
            except Exception as e:
                logger.error(f"无法解析 query: {query}, 错误: {e}")
                raise ValueError("Invalid query format")

            
        task_params = json.loads(query)
        
        print(f'task_params: {task_params}')


        task_id = 0
        coordinates = self._get_geocode(task_params["location"])
        weather = self._fetch_weather(coordinates["center"][0], coordinates["center"][1])
        start_time, end_time = self._calculate_time_range(task_params["validity_period_days"])

        return Task(task_id = task_id,
            location=task_params["location"],
            task_type="area_target" if task_params["is_area"] else "point_target",
            latitude=coordinates["center"][1],
            longitude=coordinates["center"][0],
            task_priority=task_params["task_priority"],
            time_priority=task_params["time_priority"],
            quality_priority=task_params["quality_priority"],
            validity_period=(start_time, end_time),
            area_size=task_params.get("area_size"),
            weather=weather["cloudrate"]
        )
            




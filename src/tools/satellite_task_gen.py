# -*- coding: utf-8 -*-
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List
from agno.tools import Toolkit
import logging
import uuid

class SatelliteGenTool(Toolkit):
    """卫星数据获取工具包（只负责获取原始数据）"""
    
    def __init__(self):
        super().__init__(name="Satellite Data Toolkit")
        
        # API配置
        self.amap_key = "4b80d2b289f03e3de147c8e82364eab6"
        self.caiyun_key = "67JCvZ2qyIy9B8Om"
        self.amap_geocode_url = "https://restapi.amap.com/v3/geocode/geo"
        self.weather_api_url = "https://api.caiyunapp.com/v2.5/{API_KEY}/{lon},{lat}/hourly.json"
        
        # 注册工具函数
        self.register(self.run)

    def run(self, params: dict) -> str:
        """
        任务生成工具主入口, 返回JSON字符串
        必须返回字符串类型以符合AGNO框架要求
        
        Args:
            params (dict): 输入参数字典，包含位置、天数等信息
            
        Returns:
            str: JSON字符串，包含任务ID、位置、坐标、天气、有效期等信息
        
        """
        try:
            # 参数验证并收集缺失的参数
            required = ["location", "validity_period_days"]
            missing_params = [key for key in required if key not in params]
            
            if missing_params:
                # 返回缺失参数的提示信息
                return json.dumps({
                    "error": "Missing required parameters",
                    "missing_params": missing_params,
                    "example": {
                        "location": "示例位置名称，如杭州西湖",
                        "validity_period_days": "任务有效期天数，如2",
                    }
                }, ensure_ascii=False, indent=4)
            
            
            # 0. 创建任务 UUID
            task_id = str(uuid.uuid4())
            
            # 1. 获取坐标
            geo = self._get_geocode(params["location"])
            
            # 2. 获取天气
            weather = self._get_weather(geo["lng"], geo["lat"])
            if not weather or len(weather) != 48:
                return json.dumps({
                    "error": "Failed to fetch valid weather data",
                    "details": "Weather data must contain exactly 48 values."
                }, ensure_ascii=False, indent=4)
            
            # 3. 计算有效期
            start = datetime.now()
            end = start + timedelta(days=params["validity_period_days"])
            
            # 4. 构建结果字典并转为JSON字符串
            result = {
                f"{task_id}": {
                    "location": params["location"],
                    "latitude": geo["lat"],
                    "longitude": geo["lng"],
                    "validity_period": [f"{start.isoformat()}",f"{end.isoformat()}"],
                    "cloudrate": weather,
                    "area_size": params.get("area_km", 10),
                }
               
            }
            task_gen_result = json.dumps(result, ensure_ascii=False, indent=4, separators=(",", ": "))
            
            with open("task_gen_result.json", "w", encoding="utf-8") as f:
                f.write(task_gen_result)
                # logging.info(f"Tool executed successfully: {task_gen_result}")
            
            return task_gen_result
            
        except Exception as e:
            logging.error(f"Tool execution failed: {str(e)}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    
    def _get_geocode(self, location: str) -> dict:
        """获取地理坐标"""
        url = f"https://restapi.amap.com/v3/geocode/geo?key={self.amap_key}&address={location}"
        res = requests.get(url, timeout=5).json()
        if res.get("status") == "1" and res.get("geocodes"):
            lng, lat = res["geocodes"][0]["location"].split(",")
            return {"lng": float(lng), "lat": float(lat)}
        return {"lng": 0, "lat": 0}
    

    def _get_weather(self, lon: float, lat: float) -> str:
        """调用彩云 API 获取天气"""
        url = self.weather_api_url.format(API_KEY=self.caiyun_key, lon=lon, lat=lat)
        try:
            response = requests.get(url, params={"hourlysteps": 48})
            response.raise_for_status()
            data = response.json()
            cloudrate = data.get("result", {}).get("hourly", {}).get("cloudrate", [])

            return [entry["value"] for entry in cloudrate]
        except requests.RequestException as e:

            return []  # 发生异常时返回空列表
        


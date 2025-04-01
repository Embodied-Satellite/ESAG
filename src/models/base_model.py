from datetime import datetime
from typing import List
from pydantic import BaseModel, Field, validator

class SatellitePlan(BaseModel):
    task_id: str = Field(..., description="任务编号")
    satellite_id: str = Field(..., description="卫星编号")
    position: List[float] = Field(..., description="卫星位置，格式为 [纬度, 经度]")
    observation_time: List[str] = Field(..., description="观测时间，ISO 8601 格式的时间区间")
    slew_angle: float = Field(..., description="卫星指向角度")
    solar_panel_angle: float = Field(..., description="太阳能板角度")
    status: str = Field(..., description="任务状态")
    execution_time: str = Field(..., description="任务执行时间，ISO 8601 格式")

    # 可选：添加验证器来确保 observation_time 和 execution_time 是有效的日期时间
    @classmethod
    def validate_datetime(cls, v: str) -> datetime:
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"无效的日期时间格式: {v}")

    @validator('observation_time', 'execution_time', pre=True, always=True)
    def validate_dates(cls, v):
        return [cls.validate_datetime(date) for date in v] if isinstance(v, list) else cls.validate_datetime(v)
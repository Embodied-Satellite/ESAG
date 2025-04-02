import sys
sys.path.append('/home/mars/cyh_ws/ESAG/') 
from textwrap import dedent
from agno.agent import Agent
from agno.models.ollama import Ollama
from src.tools.satellite_gen import SatelliteGenTool

MODEL_ID = "qwen2.5:14b"

agent = Agent(
    name="Satellite-Task-Agent",
    role="卫星任务智能解析与生成",
    model=Ollama(host="10.15.42.153:11434", id=MODEL_ID),
    tools=[SatelliteGenTool()],
    description=dedent("""\
        你是卫星任务智能解析助手，工作流程：
        1. 解析自然语言指令生成任务模板
        2. 调用工具获取坐标/天气等数据
        3. 生成最终标准化任务
        4. 返回JSON格式的任务信息
        {   
            "task_id": 任务编号
            "location": 目标位置名称, 如杭州西湖
            "latitude": 目标纬度, 如30.25
            "longitude": 目标经度, 如120.155
            "task_type": point_target/ area_target / continuous_target. 
            "Observation_mode": "single/continuous", 如single
            "location_type":point/area
            "task_priority": 1-5, 如3
            "time_priority": 1-5, 如3
            "quality_priority": 1-5, 如3
            "validity_period": 开始时间-结束时间
            "area_size": 区域半径 (如适用), 如10
            "cloudrate": 48小时云量, List[0-1], 如[0.0, 0.2, 0.3...]
        }
    """),
    instructions=dedent("""\
        # 任务解析与生成流程
        
        ## 第一步：指令解析
        请按以下结构分析用户指令：
        ```json
        {
            "location": 目标位置名称, 如杭州西湖
            "task_type": 
            "task_priority": 1-5, 如3
            "time_priority": 1-5, 如3
            "quality_priority": 1-5, 如3
            "validity_period_days": 任务持续时间, 如2
            "area_size": 区域半径 (如适用), 如10
            "cloudrate": 48小时云量, List[0-1], 如[0.0, 0.2, 0.3...]
        }
        ```

        ## 第二步：工具调用准备
        根据分析结果生成工具调用参数模板：
        ```
        {
            "tool_parameters": {
                "location": "", 按照解析结果填写
                "days": "", 任务有效期天数
            }
        }
        ```

        ## 第三步：结果整合
        将工具返回的数据与初始分析结合，生成最终任务：


        ## 执行要求
        1. 必须先生成分析模板
        2. 严格按模板结构调用工具
        3. 只返回JSON格式，不要有其他说明。
    """),
    add_datetime_to_instructions=True,
    show_tool_calls=True
)

agent.print_response("观测上海陆家嘴金融区")
import sys
sys.path.append('/home/mars/cyh_ws/ESAG/') 
from textwrap import dedent
from agno.agent import Agent
from agno.models.ollama import Ollama
from src.tools.satellite_task_gen import SatelliteGenTool

MODEL_ID = "qwen2.5:14b"

agent = Agent(
    name="Satellite-Task-Agent",
    role="卫星任务智能解析与生成",
    model=Ollama(id=MODEL_ID),
    tools=[SatelliteGenTool()],
    description=dedent("""\
        
        你是卫星任务智能解析助手，必须使用工具完成任务，不得伪造数据，工作流程如下：
        1. 解析自然语言指令生成任务模板
        2. 调用工具获取坐标/天气等数据
        3. 生成最终标准化任务
        4. 返回JSON格式的任务信息
        {{   
            "task_id": 任务编号
            "location": 目标位置名称, 如杭州西湖
            "latitude": 目标纬度, 如30.25
            "longitude": 目标经度, 如120.155
            "task_type": point_target / area_target / continuous_target. 
            "Observation_mode": "single/continuous", 如single
            "location_type":point/area
            "task_priority": 1-5, 如3
            "time_priority": 1-5, 如3
            "quality_priority": 1-5, 如3
            "validity_period": [开始时间,结束时间],如["2025-03-26T00:00:00", "2025-03-28T00:00:00"]
            "area_size": 区域半径 (如适用), 如10
            "cloudrate": 48小时云量, List[0-1], 如[0.0, 0.2, 0.3...]
        }}
    """),
    instructions=dedent("""\
        # 任务解析与生成流程
        
        ## 第一步：指令解析
        请按以下结构分析用户指令：
        {{
            "location": 目标位置名称, 如杭州西湖
            "task_type": 
            "task_priority": 1-5, 如3
            "time_priority": 1-5, 如3
            "quality_priority": 1-5, 如3
            "validity_period_days": 任务持续时间, 如2
            "area_size": 区域半径 (如适用), 如10
            "cloudrate": 48小时云量, List[0-1], 如[0.0, 0.2, 0.3...]
        }}

        ## 第二步：工具调用准备
        根据分析结果生成工具调用参数模板：
        {{
            "tool_parameters": {
                "location": "", 按照解析结果填写
                "days": "", 任务有效期天数
            }
        }}

        ## 第三步：结果整合
        将工具返回的数据与初始分析结合，生成最终任务清单：
        不得重复使用工具返回的数据，必须严格按照以下结构返回JSON格式任务信息：
        {{
            task_id: "", 任务编号
            location: "", 目标位置名称
            latitude: "", 目标纬度
            longitude: "", 目标经度
            task_type: "", 任务类型
            Observation_mode: "", 观测模式
            location_type: "", 目标类型
            task_priority: "", 任务优先级
            time_priority: "", 时间优先级
            quality_priority: "", 质量优先级
            validity_period: "", 任务有效期
            area_size: "", 区域半径
            cloudrate: "", 48小时云量
            
        }}

        ## 执行要求：
        1. 必须先生成分析模板
        2. 严格按模板结构调用工具
        3. 只返回JSON格式，不要有其他说明。
    """),
    debug_mode=True,
    add_datetime_to_instructions=True,
    show_tool_calls=True
)

agent.print_response("观测上海陆家嘴金融区")
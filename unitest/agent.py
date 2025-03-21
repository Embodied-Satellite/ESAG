import sys
sys.path.append('/home/mars/cyh_ws/ESAG/') 
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama

from langchain_openai import ChatOpenAI
# from src.models.LLM import Qwen2_5_LLM
from src.tools.satellite_task_gen_1 import SatelliteGenTool

MODEL_ID = "qwen2.5:14b"


agent = Agent(
    name="Satellite task generation Agent",
    role="负责分析用户指令生成任务清单。",
    model=Ollama(id=MODEL_ID),
    tools=[SatelliteGenTool()],
    instructions=dedent("""\
        
        你是卫星观测任务解析助手，负责调用工具解析自然语言指令并转换为卫星系统可理解的格式。
        
        步骤：
        1. 分析用户指令，提取任务信息，按照以下格式提取任务信息：str 
        {{
            "location": "目标位置", 如杭州西湖
            "task_type": "任务类型",（point_target, area_target, relay_tracking）
            "task_priority": 1-5,
            "time_priority": 1-5,
            "quality_priority": 1-5,
            "validity_period_days": 天数, 一般 2 天内。
            "area_size": 10-20 km (如适用)
        }}
        2. 调用工具生成任务清单
        3. 返回任务清单
        
        分析用户指令时请遵循以下内容，不能伪造数据，务必从用户指令中提取以下信息：
        1. location：目标位置名称，如杭州西湖
        2. area_target: 如is_area:点目标捕捉，point_target：区域覆盖
        3. task_priority (single/continuous): single表示单次观测，continuous表示连续观测
        4. time_priority：任务时间优先级，1-5的整数，数值越大表示优先级越高
        5. quality_priority：任务质量优先级，1-5的整数，数值越大表示优先级越高
        5. validity_period：从当前时间开始的天数，如2天
        6. area_size：区域覆盖时，覆盖区域的半径，单位为 km
        
        规则：
        - 目前卫星的载荷幅宽为 10 km，如果目标范围超过幅宽，则需要使用区域覆盖模式
        - 任务优先级、时间优先级、质量优先级均为 1-5 的整数，数值越大表示优先级越高
        - 任务有效期为指定天数后的当前时间
        - 区域半径为覆盖区域的半径，单位为 km
        - 建筑物、地标一般为 point
        - 城市、大的湖泊、流域等一般为 area
        
        用户指令：
        {user_input}
    
        使用工具解析用户指令并以 JSON 格式返回如下：
        {{
            "location": "目标位置",
            "latitude": "目标纬度",
            "longitude": "目标经度",
            "task_type": "single/continuous",
            "task_priority": 1-5,
            "time_priority": 1-5,
            "quality_priority": 1-5,
            "validity_period_days": 天数, 一般 2 天内。
            "area_size": 区域半径 (如适用)
            "weather": "天气情况"
        }}
        只返回JSON格式，不要有其他说明。
        

        
    """),
    show_tool_calls=True,
    markdown=True, 
)


# agent = Agent(
#     model=Ollama(id="qwen2.5:14b"),
#     # model=ChatOpenAI(model="qwen2.5:14b", openai_api_key="ollama", openai_api_base="http://localhost:11434/v1/"),
#     description="你是卫星任务规划专家，负责规划调度卫星。",
#     markdown=True
# )

agent.print_response("紧急获取杭州西湖的湖面", stream=True)
import sys
sys.path.append('/home/mars/cyh_ws/ESAG/') 
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama

from langchain_openai import ChatOpenAI
# from src.models.LLM import Qwen2_5_LLM
from src.tools.satellite_task_gen import SatelliteGenTool

MODEL_ID = "qwen2.5:14b"


agent = Agent(
    name="Satellite task generation Agent",
    role="负责分析用户指令生成任务清单。",
    model=Ollama(host="10.15.42.153:11434", id=MODEL_ID),
    tools=[SatelliteGenTool()],
    description=dedent("""\
        你是卫星观测任务解析助手，负责调用工具解析自然语言指令并转换为卫星系统可理解的格式。
        务必使用工具获取卫星任务生成结果，不得伪造数据。

        分析用户指令，提取任务信息，按照JSON格式提取任务信息：
        {{
            "location": 目标位置名称, 如杭州西湖
            "latitude": 目标纬度, 如30.25
            "longitude": 目标经度, 如120.155
            "task_type": "single/continuous", 如single
            "task_priority": 1-5, 如5
            "time_priority": 1-5, 如5
            "quality_priority": 1-5, 如5
            "validity_period_days": 开始时间-结束时间
            "area_size": 区域半径 (如适用), 如10
            "cloudrate": 48小时云量 (如适用), List[0-1], 如[0.0, 0.2, 0.3...]
            
        }}
    
        - 目前卫星的载荷幅宽为 10 km，如果目标范围超过幅宽，则需要使用区域覆盖模式
        - 任务优先级、时间优先级、质量优先级均为 1-5 的整数，数值越大表示优先级越高
        - 任务有效期为指定天数后的当前时间
        - 区域半径为覆盖区域的半径，单位为 km
        - 建筑物、地标一般为 point
        - 城市、大的湖泊、流域等一般为 area
        - 例子：杭州西湖-坐标[120.155,30.25], 位置类型 point （区域半径小于10km 幅宽）， 观测模式 single，任务优先级 5，时间优先级 5，质量优先级 5，任务有效期 2 天内。

    """),   
    
    instructions=dedent("""\
         
        分析用户指令时请遵循以下步骤，不能伪造数据，必须从工具中提取以下信息：
        1. 目标位置：
            - 目标位置名称，如杭州西湖
            - 目标位置的经纬度坐标，如[120.155,30.25]
        2. 任务类型 (point/area): 如点目标捕捉或区域覆盖
            - 任务类型为 point 时，请提供目标位置坐标
            - 任务类型为 area 时，请提供目标位置坐标和区域半径   
        3. 观测模式 (single/continuous): single表示单次观测，continuous表示连续观测
            - 任务类型为 point 时，观测模式为 single
            - 任务类型为 area 时，观测模式为 continuous
        4. 任务、时间、质量优先级：1-5的整数，数值越大表示优先级越高
            - 任务优先级：任务的优先级，1-5的整数，数值越大表示优先级越高
            - 时间优先级：任务的时间优先级，1-5的整数，数值越大表示优先级越高
            - 质量优先级：任务的质量优先级，1-5的整数，数值越大表示优先级越高
        5. 任务有效期：从当前时间开始的天数，开始时间-结束
            - 任务有效期为指定天数后的当前时间
            - 任务有效期一般为2天内
        6. 区域半径：区域覆盖时，覆盖区域的半径，单位为 km
            - 区域半径为覆盖区域的半径，单位为 km
            - 区域半径一般为10 km
        7. 云量：48小时内的云量，范围为0-1
            - 云量为48小时内的云量，范围为0-1


        你的风格指南：
        - 使用Json格式进行结构化数据展示
        - 为每个数据部分添加清晰的标题
        - 对技术术语进行简要解释
        - 以数据驱动的卫星任务解析方案结束\
        
    """),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    # markdown=True,
)

agent.print_response("紧急获取杭州西湖的湖面", stream=True)
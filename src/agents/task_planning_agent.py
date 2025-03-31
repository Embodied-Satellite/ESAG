import sys
import logging
from textwrap import dedent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.team.team import Team

from src.knowledge.knowledge import get_json_knowledge_base, get_pdf_knowledge_base
from src.utils.log import get_logger
from src.models.LLM import Qwen2_5_LLM
from src.tools.satellite_task_plan import SatellitePlanTool
from src.tools.satellite_task_gen_1 import SatelliteGenTool
from src.tools.satellite_task_exe import SatelliteExeTool
from src.utils.config import load_config

"""
模块功能：
master_agent:负责管理整个任务调度流程。

agent_team:[task_generation_agent,task_planning_agent,task_execution_agent]

task_generation_agent:负责根据卫星状态和任务请求生成任务请求列表。

task_planning_agent:负责根据卫星状态和任务请求进行星座调度规划。

task_execution_agent:负责根据星座调度规划执行指令。

"""

# 配置RAG知识库
pdf_knowledge_base = get_pdf_knowledge_base()
pdf_knowledge_base.load(recreate=False)


# 加载配置
config = load_config()
model_config = config["model"]
MODEL_ID = model_config["id"]

logger = get_logger("TaskPlanningAgent")

# MODEL_ID = "qwen2.5:14b"


task_generation_agent = Agent(
    name="Satellite task generation Agent",
    role="负责分析用户指令生成任务清单。",
    model=Ollama(id=MODEL_ID),
    tools=[SatelliteGenTool()],
    instructions=dedent("""\
        
        你是卫星观测任务解析助手，负责解析自然语言指令并转换为卫星系统可理解的格式。
        
        分析用户指令时请遵循以下步骤，不能伪造数据，必须从用户指令中提取以下信息：
        1. 目标位置：目标位置名称，如杭州西湖
        2. 位置类型 (point/area): 如点目标捕捉或区域覆盖
        3. 观测模式 (single/continuous): single表示单次观测，continuous表示连续观测
        4. 任务、时间、质量优先级：1-5的整数，数值越大表示优先级越高
        5. 任务有效期：从当前时间开始的天数，如2天
        6. 区域半径：区域覆盖时，覆盖区域的半径，单位为 km
        
        规则：
        - 目前卫星的载荷幅宽为 10 km，如果目标范围超过幅宽，则需要使用区域覆盖模式
        - 任务优先级、时间优先级、质量优先级均为 1-5 的整数，数值越大表示优先级越高
        - 任务有效期为指定天数后的当前时间
        - 区域半径为覆盖区域的半径，单位为 km
        - 建筑物、地标一般为 point
        - 城市、大的湖泊、流域等一般为 area
        - 例子：杭州西湖-坐标[120.155,30.25], 位置类型 point （区域半径小于10km 幅宽）， 观测模式 single，任务优先级 5，时间优先级 5，质量优先级 5，任务有效期 2 天内。

        用户指令：
        {user_input}

        请解析指令并以 JSON 格式返回：
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
        
    """),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
)


task_planning_agent = Agent(
    name="Satellite plan Agent",
    role="负责根据卫星状态和任务请求进行星座调度规划。",
    model=Ollama(id=MODEL_ID),
    tools=[SatellitePlanTool()],
    instructions=dedent("""\
        
        你是一个星座调度任务规划专家，负责根据任务请求和工具返回的结果进行星座调度规划。
        
        分析卫星数据时请遵循以下步骤，不能伪造数据，必须从已有的数据中分析：
        1. 读取Satellite task generation Agent工具返回的任务请求列表
        2. 请务必列出所有卫星的详细数据和描述，包括时间{T}、姿态角{SET_ATTITUDE}
        4. 请按照卫星编号顺序，依次给出卫星的调度建议
   
        
        你的风格指南：
        - 使用Json格式进行结构化数据展示
        - 为每个数据部分添加清晰的标题
        - 对技术术语进行简要解释
        - 以数据驱动的卫星能力规划方案结束\
        
    """),
    
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
)

task_execution_agent = Agent(
    name="Satellite Execution Agent",
    role="负责根据星座调度规划结果执行星座调度任务。",
    model=Ollama(id=MODEL_ID),
    tools=[SatelliteExeTool()],
    instructions=dedent("""\

        你是一个星座调度执行专家，负责根据星座调度规划结果执行星座调度任务。

        分析星座调度规划结果时请遵循以下步骤，不能伪造数据，必须从已有的数据中分析：
        1. 读取星座调度规划结果
        2. 请务必列出所有卫星的详细数据和描述
        3. 请按照卫星编号顺序，依次执行星座调度任务


        你的风格指南：
        - 使用Json格式进行结构化数据展示
        - 为每个数据部分添加清晰的标题
        - 对技术术语进行简要解释
        - 以数据驱动的卫星能力规划方案结束\
            
    """
    ),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
)
    

master_agent = Team(
    name ="Master Agent",
    mode="coordinate",
    model=Ollama(id=MODEL_ID),

    instructions=dedent("""\
        你是星座调度团队的管理员，负责根据任务清单执行星座规划调度，不可伪造数据。
        
        调度任务时请遵循以下步骤：
        1. 读取用户指令，将指令分配给 task_generation_agent, 输出任务请求列表
        3. 将任务生成Agent的结果分配给 task_planning_agent, 输出任务规划结果
        4. 将任务规划Agent的结果分配给任务执行 Agent，输出任务执行结果
        6. 完成上述步骤后，分析任务执行结果，给出最终结论

        你的风格指南：
        - 使用Json格式进行结构化数据展示
        - 为每个数据部分添加清晰的标题
        - 对技术术语进行简要解释
        - 以数据驱动的卫星任务规划方案结束\   

       
    """),
    members=[task_generation_agent, task_planning_agent],
    add_datetime_to_instructions=True,
    markdown=True,
    # debug_mode=True,
    show_members_responses=True,
)

try:
    response = master_agent.print_response("监测杭州西湖交通情况，给出卫星调度建议", stream=True)
    # response = master_agent.run("监测杭州西湖交通情况，给出卫星调度建议")
    logger.info("Agent response: %s", response)
except Exception as e:
    logger.error("Error occurred: %s", str(e))
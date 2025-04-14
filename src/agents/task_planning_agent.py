import sys
import logging

from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.models.huggingface import HuggingFace
from agno.team.team import Team
from src.knowledge.knowledge import get_json_knowledge_base, get_pdf_knowledge_base
from src.utils.log import get_logger
# from src.models.LLM import Qwen2_5_LLM
from src.tools.satellite_task_plan import SatellitePlanTool
from src.tools.satellite_task_gen import SatelliteGenTool
from src.tools.satellite_task_exe import SatelliteExeTool
from src.utils.config import load_config
from src.models.base_model import SatellitePlan


"""
模块功能：
master_agent:负责管理整个任务调度流程。

agent_team:[task_generation_agent,task_planning_agent,task_execution_agent]

task_generation_agent:负责根据卫星状态和任务请求生成任务请求列表。

task_planning_agent:负责根据卫星状态和任务请求进行星座调度规划。

task_execution_agent:负责根据星座调度规划执行指令。

"""

# 配置RAG知识库
# pdf_knowledge_base = get_pdf_knowledge_base().load(recreate=True)
# json_knowledge_base = get_json_knowledge_base().load(recreate=True)

# 加载config配置
config = load_config()
model_config = config["ollama_model"]
MODEL_HOST = model_config["host"]
MODEL_ID = model_config["id"]
TEMPERATURE = model_config["temperature"]

logger = get_logger("TaskPlanningAgent")

task_generation_agent = Agent(
    
    name="Satellite-TaskGen-Agent",
    role="负责用户指令解析与卫星任务生成",
    model=Ollama(id=MODEL_ID),
    tools=[SatelliteGenTool()],
    description=dedent("""\
        
        你是卫星任务智能解析助手，必须使用工具完成任务，不得伪造数据，工作流程如下：
        1. 解析自然语言指令生成任务模板
        2. 调用工具获取坐标/天气等数据
        3. 生成最终标准化任务
        4. 返回JSON格式的任务信息，格式如下：
        
        {{   
            "task_id": 任务编号
            "location": 目标位置名称, 如杭州西湖
            "latitude": 目标纬度, 如30.25
            "longitude": 目标经度, 如120.155
            "task_type": point_target / area_target / continuous_target. 
            "Observation_mode": "single/continuous", 如single
            "task_priority": 1-5, 如3
            "time_priority": 1-5, 如3
            "quality_priority": 1-5, 如3
            "validity_period": 任务有效期
            "area_size": 区域半径
            "cloudrate": 48小时云量
        }}
        
        - 任务编号为UUID格式,不可伪造
        - 任务类型为point_target / area_target / continuous_target
        - 观测模式为single / continuous
        - 任务优先级、时间优先级、质量优先级为1-5
        - 任务有效期格式为开始时间-结束时间
        - 区域半径默认为10km
        - 云量范围为48个0-1的浮点数列表，如[0.0, 0.2, 0.3...]\
        
    """),
    instructions=dedent("""\
        # 任务解析与生成流程
        
        ## 第一步：指令解析
        请按以下结构分析用户指令：
        {{
            "location": 目标位置名称, 如杭州西湖
            "task_type": 任务类型 (point_target/area_target/continuous_target)
            "task_priority": 1-5, 如3
            "time_priority": 1-5, 如3
            "quality_priority": 1-5, 如3
            "validity_period_days": 任务持续时间, 如2
            "area_size": 区域半径, 默认为10
            "cloudrate": 48个0-1的浮点数列表，如[0.0, 0.2, 0.3...]\
        }}

        ## 第二步：工具调用准备
        根据分析结果生成工具调用参数模板：
        ```
        {
            "tool_parameters": {
                "location": "", 按照解析结果填写
                "validity_period_days": "", 任务持续时间天数
            }
        }
        ```

        ## 第三步：结果整合
        将工具返回的数据与初始分析结合，生成最终任务清单：
        必须严格按照以下结构返回JSON格式任务信息:
        
        {{
            {
                task_id:"", 工具返回的任务编号
                location: "", 工具返回的目标位置名称
                latitude: "", 工具返回的目标纬度
                longitude: "", 工具返回的目标经度
                task_type: "", 工具返回的任务类型
                Observation_mode: "", 工具返回的观测模式
                task_priority: "", 任务优先级
                time_priority: "", 时间优先级
                quality_priority: "", 质量优先级
                validity_period: "", 任务有效期
                area_size: "", 工具返回的区域半径
                cloudrate: "", 工具返回的48小时云量，List[0-1]
            }
            
        }}

        ## 执行要求：
        1. 必须先生成分析模板
        2. 严格按模板结构调用工具
        3. 只返回JSON格式，不要有其他说明。\
    """),
    # debug_mode=True,
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    use_json_mode=True,
    
)


task_planning_agent = Agent(
    name="Satellite-TaskPlan-Agent",
    role="负责调用工具进行星座调度规划。",
    model=Ollama(id=MODEL_ID),
    tools=[SatellitePlanTool()],
    description=dedent("""\
        
        你是一个星座调度任务规划助手，负责根据任务请求和工具返回的结果进行星座调度规划。
        
        任务规划必须使用工具，并遵循以下步骤，不能伪造数据：
        1. 读取任务请求列表
        2. 根据任务请求生成工具调用参数模板
        2. 调用工具获取卫星状态数据
        3. 根据卫星状态数据和任务请求进行星座调度规划
        4. 返回JSON格式的星座调度规划结果，格式如下：
        
        {{
            "task_id": 任务编号, 与 Satellite-TaskGen-Agent 工具返回一致
            "satellite_id": 卫星编号，如"Satellite_4",
            "observation_time":执行时间，Satellite-TaskGen-Agent 工具返回的执行时间
            "slew_angle": 卫星测摆角度，如15,
            "solar_panel_angle": （若适用）太阳能帆板角度，如45
        }}
        
        - 卫星名称为卫星的唯一标识符，不可伪造
        - 执行时间为卫星执行任务的时间
        - 测摆角度为卫星在轨道上的测摆角度\
        
    """),
    
    instructions=dedent("""\
        
        # 任务清单规划流程
        
        ## 第一步：任务解析
        请按以下结构分析任务清单：
        {{
            "task_id": 任务编号
            "location": 目标位置名称
            "latitude": 目标纬度, 如30.25
            "longitude": 目标经度, 如120.155
            "task_type": point_target / area_target / continuous_target.
            "Observation_mode": "single/continuous", 如single
            "task_priority": 1-5, 如3
            "time_priority": 1-5, 如3
            "quality_priority": 1-5, 如3
            "validity_period": 任务有效期,如[开始时间,结束时间]
            "area_size": 区域半径, 默认为10km
            "cloudrate": 48小时云量, List[0-1]
        }}

        ## 第二步：工具调用
        根据分析结果生成工具调用参数模板：
        {{
            "tool_parameters": {
                "task_id": "", 任务编号与 Satellite-TaskGen-Agent 工具返回一致
                "observation_time": "", 工具返回的执行时间

            }
        }}

        ## 第三步：结果整合
        将工具返回的数据与初始分析结合，生成星座调度规划结果。
        星座规划结果必须严格按照以下结构返回JSON格式:
        {{
            "task_id": 任务编号, 与 Satellite-TaskGen-Agent 工具返回一致
            "satellite_id": "Satellite_4"
            "observation_time":工具返回的执行时间
            "slew_angle": 工具返回的卫星测摆角度,如15
            "solar_panel_angle": (若适用)太阳能帆板角度,如45
        }}
        
        ## 执行要求：
            1. 必须从工具中获取 task_id 和 observation_time。
            2. 确保时间数据格式正确且范围合理。
            3. 返回的结果必须是 JSON 格式，不能包含其他说明。\
    """),
    
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    use_json_mode=True,
)


task_execution_agent = Agent(
    name="Satellite-TaskExe-Agent",
    role="负责根据星座调度规划结果执行星座调度任务。",
    model=Ollama(id=MODEL_ID),
    tools=[SatelliteExeTool()],
    instructions=dedent("""\

        你是一个星座调度执行专家，负责根据星座调度规划结果执行星座调度任务。

        分析星座调度规划结果时请遵循以下步骤，不能伪造数据，必须从已有的数据中分析：
        1. 读取星座调度规划结果
        2. 请务必列出所有卫星的详细数据和描述
        3. 根据卫星状态数据和任务请求进行星座调度规划

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
    

master_agent = Team(
    name="Satellite Planning Agent Team",
    mode="coordinate",
    model=Ollama(id=MODEL_ID),
    # knowledge=json_knowledge_base,
    members=[task_generation_agent, task_planning_agent, task_execution_agent],
    description=dedent("""\
        你是星座调度团队协调员，负责分配任务给三个团队成员，并确保任务按顺序完成。每个成员的角色如下：
        1. `Satellite-TaskGen-Agent`：负责解析用户指令并生成任务请求列表。
        2. `Satellite-TaskPlan-Agent`：负责根据读取的JSON文件行星座调度规划。
        3. `Satellite-TaskExe-Agent`：负责根据星座调度规划结果执行任务。
        请确保每个成员使用工具完成任务，且输出符合预期格式，并严格按照顺序执行任务。\
    """),
    
    instructions=dedent("""\
        
        你必须协调 {members} 中的团队成员按顺序完成任务，确保每个步骤的结果正确且完整。以下是任务调度流程：
        
        # 任务调度流程
        ## 第一步：分配任务请求助手
        将用户指令分配给任务助手 `Satellite-TaskGen-Agent`，使用工具生成任务请求列表。任务助手必须返回以下 JSON 格式的任务请求列表：
        ```
        [
            {
                "task_id": "任务编号",
                "location": "目标位置名称",
                "latitude": "目标纬度",
                "longitude": "目标经度",
                "task_type": "任务类型 (point_target/area_target/continuous_target)",
                "Observation_mode": "观测模式 (single/continuous)",
                "task_priority": "任务优先级 (1-5)",
                "time_priority": "时间优先级 (1-5)",
                "quality_priority": "质量优先级 (1-5)",
                "validity_period": ["开始时间", "结束时间"],
                "area_size": "区域半径 (10km)",
                "cloudrate": "48小时云量,工具返回的48小时云量，List[0-1]"
            }
        ]
        ```

        ## 第二步：分配任务规划助手
        将任务请求列表分配给规划助手 `Satellite-TaskPlan-Agent`，使用工具进行星座调度规划。规划助手必须返回以下 JSON 格式的星座调度规划结果：
        ```
        [
            {
                "task_id": "任务编号",
                "satellite_id": "卫星编号",
                "observation_time": ["开始时间", "结束时间"],
                "slew_angle": "卫星测摆角度",
                "solar_panel_angle": "太阳能帆板角度"
            }
        ]
        ```

        ## 第三步：分配任务执行助手
        将星座调度规划结果分配给执行助手 `Satellite-TaskExe-Agent`，分析任务执行结果并输出最终结论。执行助手必须返回以下 JSON 格式的执行结果：
        ```
        {
            "task_id": "任务编号",
            "execution_status": "执行状态 (success/failure)",
            "details": "执行详情，包括卫星状态和任务完成情况"
        }

        # 执行要求：
        1. 必须严格按照顺序执行三个步骤，不能跳过任何步骤。
        2. 每个步骤的输入必须是上一步的输出，不能伪造数据。
        3. 如果某一步失败，必须停止后续步骤并返回错误信息。
        4. 所有返回结果必须是 JSON 格式，不能包含其他说明。\
    """),
    
    # read_team_history=True,
    
    add_datetime_to_instructions=True,
    enable_agentic_context=True,
    share_member_interactions=True,
    show_members_responses=True,
    debug_mode=True,
    markdown=True,
)
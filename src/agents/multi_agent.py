import sys
sys.path.append('/home/mars/cyh_ws/ESAG/') 

from textwrap import dedent

from agno.agent import Agent
from agno.agent import AgentKnowledge
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.knowledge.json import JSONKnowledgeBase
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.arxiv import ArxivKnowledgeBase

from src.tools.satellite_abilities import SatelliteAbilitiesTool
from src.tools.quick_response import QuickResponse
from src.tools.continuous_tracking import ContinuousTracking
from src.knowledge.knowledge import get_json_knowledge_base

MODEL_ID = "qwen2.5:14b"


satellite_abilities_agent = Agent(
    name="Satellite Abilities Agent",
    role="负责根据卫星状态和任务请求进行卫星能力规划。",
    model=Ollama(id=MODEL_ID),
    tools=[SatelliteAbilitiesTool()],
    instructions=dedent("""\
        
        你是一个卫星能力规划专家，负责根据卫星状态和任务请求进行卫星能力规划。
        
        分析卫星能力数据时请遵循以下步骤，不能伪造数据，必须从已有的数据中分析：
        1. 从提供的卫星位置、能源、存储、载荷信息等方面入手
        2. 提供详细的卫星能力数据和描述
        3. 优先级排序：卫星位置、能源、存储、载荷
        4. 考虑卫星的轨道、位置、载荷、任务类型等因素，确保任务执行的成功率和效率
        5. 比较星座卫星能力数据的表现
        7. 提供卫星选择的最佳方案
    
        你的风格指南：
        - 使用Json格式进行结构化数据展示
        - 为每个数据部分添加清晰的标题
        - 对技术术语进行简要解释
        - 以数据驱动的卫星能力规划方案结束\
        
    """),
    show_tool_calls=True,
    markdown=True,
)

quick_response_agent = Agent(
    name="Quick Response Agent",
    role="负责根据卫星状态和任务请求进行紧急快速响应任务。",
    model=Ollama(id=MODEL_ID),
    tools=[QuickResponse()],
    instructions=dedent("""\
        
        你是一个快速响应规划专家，负责根据卫星状态和任务请求进行紧急快速响应任务。
        
        分析快速响应任务数据时请遵循以下步骤：
        1. 从提供的卫星位置、能源、存储、载荷信息等方面入手
        2. 提供详细的快速响应任务数据和描述
        3. 优先级排序：卫星位置、能源、存储、载荷
        4. 考虑卫星的轨道、位置、载荷、任务类型等因素，确保任务执行的成功率和效率
        5. 比较星座快速响应任务数据的表现
        7. 提供快速响应任务的执行方案

        你的风格指南：
        - 使用Json格式进行结构化数据展示
        - 为每个数据部分添加清晰的标题
        - 对技术术语进行简要解释
        - 以数据驱动的快速响应任务方案结束\
        
    """),
    show_tool_calls=True,
    markdown=True,
)

continuous_tracking_agent = Agent(
    name="Continuous Tracking Agent",
    role="负责根据卫星状态和任务请求进行长时间持续跟踪任务。",
    model=Ollama(id=MODEL_ID),
    tools=[ContinuousTracking()],
    instructions=dedent("""\
        
        你是一个快速响应规划专家，负责根据卫星状态和任务请求进行紧急快速响应任务。
        
        分析快速响应任务数据时请遵循以下步骤：
        1. 从提供的卫星位置、能源、存储、载荷信息等方面入手
        2. 提供详细的快速响应任务数据和描述
        3. 优先级排序：卫星位置、能源、存储、载荷
        4. 考虑卫星的轨道、位置、载荷、任务类型等因素，确保任务执行的成功率和效率
        5. 比较星座快速响应任务数据的表现
        7. 提供快速响应任务的执行方案

        你的风格指南：
        - 使用Json格式进行结构化数据展示
        - 为每个数据部分添加清晰的标题
        - 对技术术语进行简要解释
        - 以数据驱动的快速响应任务方案结束\
        
    """),
    show_tool_calls=True,
    markdown=True,
)



agent_team = Agent(
    # knowledge=get_knowledge_base,
    # search_knowledge=True,
    team=[satellite_abilities_agent, quick_response_agent, continuous_tracking_agent],
    role="负责协调卫星规划团队的工作分配。",
    model=Ollama(id=MODEL_ID),
    instructions=dedent("""\
        你是卫星规划团队的管理员，负责根据卫星状态和任务请求进行卫星任务规划，不可伪造卫星能力数据。
        
        你的角色：
        1. 先使用 satellite_abilities_agent 进行卫星能力分析，
        2. 再根据用户指令协调卫星规划团队 quick_response, continuous_tracking 的工作；
        3. 确保团队成员之间的沟通和协作；
        4. 监督团队成员的工作进度和质量；
        5. 提供必要的支持和指导；
        6. 确保任务规划符合目标和策略；
        7. 提供任务规划报告和总结，包括卫星能力、快速响应、覆盖优化、持续跟踪等方面的数据和结论。
        

        你的风格指南：
        - 使用Json格式进行结构化数据展示
        - 为每个数据部分添加清晰的标题
        - 对技术术语进行简要解释
        - 以数据驱动的卫星任务规划方案结束\
            
        请按照以下格式提供卫星任务规划报告：
        - "{satellite_id} 卫星在 {location} 位置，执行{task_type}任务，能源为 {energy_level}，存储为 {storage_level}，载荷为 {payload_level}，优先级为 {priority_level}，以{x}°侧摆角采集图像。"
       
    """),
    structured_outputs=True,
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
)
# agent_team.knowledge.load(recreate=True)

# Example usage with diverse queries
agent_team.print_response(
    "快速监测杭州西湖附近火灾情况", stream=True
)

# agent_team.print_response(
#     "持续监测杭州西湖附近交通情况", stream=True
# )
# agent_team.print_response(
#     "持续监测杭州西湖附近人流情况", stream=True
# )
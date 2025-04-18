import sys
sys.path.append('/home/mars/cyh_ws/ESAG/') 
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama

from langchain_openai import ChatOpenAI
# from src.models.LLM import Qwen2_5_LLM
from src.tools.satellite_task_plan import SatellitePlanTool
from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.embedder.ollama import OllamaEmbedder


MODEL_ID = "qwen2.5:7b-instruct"


knowledge_base = JSONKnowledgeBase(
    path="task_gen_result.json",
    # Table name: ai.json_documents
    vector_db=PgVector(
        table_name="json_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=OllamaEmbedder(id=MODEL_ID, dimensions=3072)
    ),
)

# knowledge_base.load(recreate=True)

task_planning_agent = Agent(
 name="Satellite-TaskPlan-Agent",
    role="负责根据任务清单调用工具进行星座调度规划。",
    model=Ollama(id=MODEL_ID),
    tools=[SatellitePlanTool()],
    # knowledge=knowledge_base,
    description=dedent("""\
        
        你是一个星座调度任务规划助手，负责分析task_execute.json文件进行星座调度规划。
        
        分析卫星数据时请必须使用工具，并遵循以下步骤，不能伪造数据：
        1. 读取json文件task_execute.json
        2. 返回JSON格式的星座调度规划结果
        {{
            "task_id": 任务编号, 与 Satellite-TaskGen-Agent 工具返回一致
            "satellite_id": 卫星编号，如"Satellite_4",
            "observation_time":执行时间，如 ["2025-03-26T10:00:00", "2025-03-26T10:15:00"],
            "slew_angle": 测摆角度，如15,
            "solar_panel_angle": （若适用）太阳能帆板角度，如45
        }}
        
        - 卫星名称为卫星的唯一标识符，不可伪造
        - 执行时间为卫星执行任务的时间
        - 测摆角度为卫星在轨道上的测摆角度\
        
    """),
    
    instructions=dedent("""\
        
        # 任务清单规划流程
        
        ## 第一步：任务解析
        读取任务清单文件task_execute.json，并解析任务请求列表。
        
        ## 第二步：结果整合
        将工具返回的数据与初始分析结合，生成星座调度规划结果。
        星座规划结果必须严格按照以下结构返回JSON格式:
        {{
            "task_id": 任务编号
            "satellite_id": 工具返回的卫星编号，如"Satellite_4",
            "observation_time":工具返回的执行时间，如 ["2025-03-26T10:00:00", "2025-03-26T10:15:00"],
            "slew_angle": 工具返回的测摆角度,如15,
            "solar_panel_angle": (若适用)太阳能帆板角度,如45
        }}
        
        ## 执行要求：
        1. 必须先生成规划模板
        2. 严格按模板结构调用工具
        3. 只返回JSON格式,不要有其他说明。
    """),
    
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    use_json_mode=True,
    debug_mode=True,
)

try:
    task_planning_agent.print_response("统计新疆哈密地区附近森林面积", stream=True)
except Exception as e:
    import traceback
    print("任务调度失败，错误信息如下：")
    traceback.print_exc()
import sys
sys.path.append('/home/mars/cyh_ws/agno/') 

from agno.agent import Agent
# from src.knowledge.knowledge import knowledge_base
from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.embedder.ollama import OllamaEmbedder
from agno.models.ollama import Ollama

from agno.knowledge.website import WebsiteKnowledgeBase
from agno.vectordb.pgvector import PgVector


knowledge_base = JSONKnowledgeBase(
    path="task_knowledge.json",
    # Table name: ai.json_documents
    vector_db=PgVector(
        table_name="json_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=OllamaEmbedder(id="llama3.2", dimensions=3072)
    ),
)

knowledge_base.load(recreate=True)

agent = Agent(
    knowledge=knowledge_base,
    model=Ollama(id="qwen2.5:14b"),
    search_knowledge=True,
    markdown=True,
)

agent.print_response("监测杭州西湖附近的交通状况", stream=True)
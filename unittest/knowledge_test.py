import sys
sys.path.append('/home/mars/cyh_ws/agno/') 

from agno.agent import Agent
# from src.knowledge.knowledge import knowledge_base
from agno.knowledge.docx import DocxKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.embedder.ollama import OllamaEmbedder
from agno.models.ollama import Ollama



knowledge_base = DocxKnowledgeBase(
    path="./knowledge.doc",
    # Table name: ai.docx_documents
    vector_db=PgVector(
        table_name="docx_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=OllamaEmbedder(id="qwen2.5:14b")
    ),
)

agent = Agent(
    knowledge=knowledge_base,
    model=Ollama(id="qwen2.5:14b"),
    search_knowledge=True,
)
agent.knowledge.load(recreate=False)

agent.print_response("遥感卫星对地观测任务的调度优先级是什么？", stream=True)
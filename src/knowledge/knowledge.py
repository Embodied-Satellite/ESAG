from agno.knowledge.json import JSONKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.embedder.ollama import OllamaEmbedder
from agno.vectordb.pgvector import PgVector
from src.utils.config import load_config

# 加载配置
config = load_config()
db_config = config["database"]
file_paths = config["file_paths"]
model_config = config["model"]

def get_json_knowledge_base():
    knowledge_base = JSONKnowledgeBase(
        path=file_paths["knowledge_base_path"],
        vector_db=PgVector(
            table_name=db_config["json_table"],
            db_url=db_config["db_url"],
            embedder=OllamaEmbedder(id=model_config["id"])
        ),
    )
    return knowledge_base

def get_pdf_knowledge_base():
    pdf_knowledge_base = PDFKnowledgeBase(
        path=file_paths["knowledge_base_path"],
        vector_db=PgVector(
            table_name=db_config["pdf_table"],
            db_url=db_config["db_url"],
            embedder=OllamaEmbedder(id=model_config["id"])
        ),
        reader=PDFReader(chunk=True),
    )
    return pdf_knowledge_base
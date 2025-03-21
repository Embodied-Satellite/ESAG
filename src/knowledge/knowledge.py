from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.pgvector import PgVector

MODEL_ID = "qwen2.5:14b"
FILE_PATH = '/home/mars/cyh_ws/SPA/satellite_data.json'
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

def get_json_knowledge_base():
    
    knowledge_base = JSONKnowledgeBase(
        path=FILE_PATH,
        # Table name: ai.json_documents
        vector_db=PgVector(
            table_name="json_documents",
            db_url=db_url,
            embedder=OllamaEmbedder(id=MODEL_ID)
        ),
    )  
    return knowledge_base

def get_pdf_knowledge_base():
    
    pdf_knowledge_base = PDFKnowledgeBase(
        path=FILE_PATH,
        # Table name: ai.pdf_documents
        vector_db=PgVector(
            table_name="pdf_documents",
            db_url=db_url,
            embedder=OllamaEmbedder(id=MODEL_ID)
        ),
        reader=PDFReader(chunk=True),
    )
    return pdf_knowledge_base


def get_doc_knowledge_base():
    
    knowledge_base = DocxKnowledgeBase(
        path=FILE_PATH,
        # Table name: ai.docx_documents
        vector_db=PgVector(
            table_name="docx_documents",
            db_url=db_url,
            embedder=OllamaEmbedder(id=MODEL_ID)
        ),
    )
    return knowledge_base

def get_csv_knowledge_base():
    
    knowledge_base = CSVKnowledgeBase(
        path=FILE_PATH,
        # Table name: ai.csv_documents
        vector_db=PgVector(
            table_name="csv_documents",
            db_url=db_url,
            embedder=OllamaEmbedder(id=MODEL_ID)
        ),
    )
    return knowledge_base
        
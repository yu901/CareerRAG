from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.main.python.qa import CareerRAG
import uvicorn

app = FastAPI(
    title="CareerRAG API",
    description="채용공고 기반 RAG 시스템 API",
    version="0.1.0",
)

# RAG 시스템 인스턴스. API 시작 시 로드됩니다.
rag_system = None

@app.on_event("startup")
def startup_event():
    """API 시작 시 RAG 시스템을 초기화합니다."""
    global rag_system
    try:
        rag_system = CareerRAG()
        print("CareerRAG system loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run the data collection and embedding pipeline first.")
        rag_system = None 
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")
        rag_system = None

class QueryRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list

@app.get("/health")
def health_check():
    """API 서버와 RAG 시스템의 상태를 확인합니다."""
    if rag_system:
        return {"status": "ok", "rag_system": "loaded"}
    return {"status": "ok", "rag_system": "not_loaded"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QueryRequest):
    """RAG 시스템에 질문하고 답변을 받습니다."""
    if not rag_system:
        raise HTTPException(
            status_code=503, 
            detail="RAG system is not available. Please check server logs."
        )
    
    try:
        result = rag_system.ask(request.query)
        
        sources = [
            {
                "title": doc.metadata.get('cleaned_title', 'N/A'),
                "company": doc.metadata.get('company', 'N/A'),
                "link": doc.metadata.get('link', 'N/A')
            }
            for doc in result["source_documents"]
        ]
        
        return AnswerResponse(answer=result["result"], sources=sources)

    except Exception as e:
        print(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the query.")

if __name__ == "__main__":
    # FastAPI 서버 실행
    print("Starting FastAPI server...")
    print("API documentation available at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

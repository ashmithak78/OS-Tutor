"""
app.py
======
FastAPI Backend for OS-Tutor.

Wraps the ChromaDB retrieval, intent routing, and generation logic into clean REST endpoints:
- POST /ask: Takes question, returns answer with exact citations and detected mode
- POST /generate-question: Takes topic and mode (mcq / lab / viva), returns generated question
- GET /health: Health status and system configuration
- GET /: Serves the interactive OS-Tutor student chat frontend
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# Re-use the existing ChromaDB and model generation logic from ask_finetuned
from ask_finetuned import answer_question, retrieve_context, classify_intent

app = FastAPI(
    title="OS-Tutor API",
    description="AI Teaching Assistant for Operating Systems (RAG + QLoRA Fine-Tuning)",
    version="1.0.0"
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Schemas
class AskRequest(BaseModel):
    question: str = Field(..., example="What is a process?")
    mode: Optional[str] = Field("auto", example="auto", description="Mode: auto, explain, mcq, lab, viva")
    use_ollama: Optional[bool] = Field(True, description="Whether to use local Ollama model for fast generation")

class Citation(BaseModel):
    file: str
    page: int
    distance: float

class AskResponse(BaseModel):
    question: str
    answer: str
    mode: str
    found: bool
    citations: List[Citation]
    distance: float

class GenerateQuestionRequest(BaseModel):
    topic: str = Field(..., example="process states")
    mode: str = Field(..., example="mcq", description="Mode: mcq, lab, or viva")
    use_ollama: Optional[bool] = Field(True, description="Whether to use local Ollama model")

class GenerateQuestionResponse(BaseModel):
    topic: str
    mode: str
    generated_content: str
    found: bool
    citations: List[Citation]
    distance: float

@app.get("/health")
def health_check():
    """Health check endpoint displaying system readiness."""
    return {
        "status": "healthy",
        "service": "OS-Tutor API",
        "version": "1.0.0",
        "chroma_collection": "os_notes",
        "supported_modes": ["explain", "mcq", "lab", "viva"]
    }

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest):
    """
    Primary Q&A endpoint.
    Retrieves context, verifies similarity threshold, classifies intent,
    and returns cited answer.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    selected_mode = None if payload.mode == "auto" else payload.mode
    result = answer_question(
        payload.question,
        mode=selected_mode,
        use_ollama_fallback=payload.use_ollama
    )

    return AskResponse(
        question=payload.question,
        answer=result["answer"],
        mode=result["mode"],
        found=result["found"],
        citations=[Citation(**c) for c in result["citations"]],
        distance=result["distance"]
    )

@app.post("/generate-question", response_model=GenerateQuestionResponse)
def generate_question_endpoint(payload: GenerateQuestionRequest):
    """
    Question generation endpoint.
    Generates MCQs, Lab exercises, or Viva questions for a given topic.
    """
    mode = payload.mode.lower()
    if mode not in ["mcq", "lab", "viva", "explain"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{payload.mode}'. Must be one of: mcq, lab, viva, explain."
        )

    prompt = f"Generate {mode} on {payload.topic}"
    result = answer_question(
        prompt,
        mode=mode,
        use_ollama_fallback=payload.use_ollama
    )

    return GenerateQuestionResponse(
        topic=payload.topic,
        mode=result["mode"],
        generated_content=result["answer"],
        found=result["found"],
        citations=[Citation(**c) for c in result["citations"]],
        distance=result["distance"]
    )

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serves the minimal, elegant OS-Tutor student chat interface."""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>OS-Tutor Backend API is running! Access /docs for Swagger UI.</h1>"

if __name__ == "__main__":
    import uvicorn
    print("Starting OS-Tutor FastAPI server on http://localhost:8000...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

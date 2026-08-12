from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import run_research


app = FastAPI(
    title="Nova Flash 1 API",
    description="Research agent API for Nova Flash 1",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://nova-flash-psi.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    query: str
    user_id: str = "default"
    session_id: str = "default"

class ResearchResponse(BaseModel):
    response: str

@app.get("/")
def home():
    return {
        "status": "Nova Flash 1 Agent Running",
        "version": "1.0.0",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post(
    "/research",
    response_model=ResearchResponse,
)
def research(request: ResearchRequest):

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Research query cannot be empty.",
        )

    try:

        response = run_research(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
        )

        return {
            "response": response,
        }

    except Exception as e:

        print("Nova Error:", e)

        raise HTTPException(
            status_code=500,
            detail="Nova encountered an error while processing the request.",
        )
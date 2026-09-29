from fastapi import FastAPI
from pydantic import BaseModel
from app.embedding_model import calculate_embedding

app = FastAPI()


class EmbeddingRequest(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "Embedding API is running"}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.text)

    return {
        "text": request.text,
        "embedding": embedding.tolist()
    }
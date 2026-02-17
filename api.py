from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from matcher import match_names


app = FastAPI(title="Name Similarity API", version="1.0.0")


class MatchRequest(BaseModel):
    name_1: str = Field(..., min_length=1)
    name_2: str = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match")
def match(payload: MatchRequest):
    name_1 = payload.name_1.strip()
    name_2 = payload.name_2.strip()

    if not name_1 or not name_2:
        raise HTTPException(status_code=400, detail="Both name_1 and name_2 are required.")

    return match_names(name_1, name_2)

from pydantic import BaseModel
from typing import List


class PostGet(BaseModel):
    """
    "id": 345,
    "text": "COVID-19 runs wild....",
    "topic": "news
    """
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    """
    "exp_group": "control",
    "recommendations": PostGet
    """
    exp_group: str
    recommendations: List[PostGet]

from typing import List, Optional
from pydantic import BaseModel


class Service(BaseModel):
    id: int
    category: str
    title: str
    description: str
    price: int
    embedding: Optional[List[float]] = None


class Kursus(BaseModel):
    id: int
    course_name: str
    level: str
    description: str
    price: int
    start_date: str
    end_date: str
    embedding: Optional[List[float]] = None


class Faq(BaseModel):
    id: int
    category: str
    title: str
    description: str
    embedding: Optional[List[float]] = None

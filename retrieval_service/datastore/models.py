from typing import List, Optional


class Service:
    def __init__(
        self,
        id: int,
        category: str,
        title: str,
        description: str,
        price: int,
        embedding: Optional[List[float]] = None,
    ):
        self.id = id
        self.category = category
        self.title = title
        self.description = description
        self.price = price
        self.embedding = embedding


class Kursus:
    def __init__(
        self,
        id: int,
        course_name: str,
        level: str,
        description: str,
        price: int,
        start_date: str,
        end_date: str,
        embedding: Optional[List[float]] = None,
    ):
        self.id = id
        self.course_name = course_name
        self.level = level
        self.description = description
        self.price = price
        self.start_date = start_date
        self.end_date = end_date
        self.embedding = embedding


class Faq:
    def __init__(
        self,
        id: int,
        category: str,
        title: str,
        description: str,
        embedding: Optional[List[float]] = None,
    ):
        self.id = id
        self.category = category
        self.title = title
        self.description = description
        self.embedding = embedding

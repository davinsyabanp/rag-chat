# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from datetime import datetime
from ipaddress import IPv4Address, IPv6Address
from typing import Any, Literal, Optional

import asyncpg
from pgvector.asyncpg import register_vector
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

import models

from .. import datastore
from ..helpers import format_sql

POSTGRES_IDENTIFIER = "postgres"


class Config(BaseModel, datastore.AbstractConfig):
    kind: Literal["postgres"]
    host: IPv4Address | IPv6Address = IPv4Address("127.0.0.1")
    port: int = 5432
    user: str
    password: str
    database: str


class Client(datastore.Client[Config]):
    __async_engine: AsyncEngine

    @datastore.classproperty
    def kind(cls):
        return POSTGRES_IDENTIFIER

    def __init__(self, async_engine: AsyncEngine):
        self.__async_engine = async_engine

    @classmethod
    async def create(cls, config: Config) -> "Client":
        async def getconn() -> asyncpg.Connection:
            conn: asyncpg.Connection = await asyncpg.connection.connect(
                host=str(config.host),
                user=config.user,
                password=config.password,
                database=config.database,
                port=config.port,
            )
            await register_vector(conn)
            return conn

        async_engine = create_async_engine(
            "postgresql+asyncpg://",
            async_creator=getconn,
        )
        if async_engine is None:
            raise TypeError("async_engine not instantiated")
        return cls(async_engine)

    async def initialize_data(
        self,
        services: list[models.Service],
        kursus_list: list[models.Kursus],
        faqs: list[models.FAQ],
    ) -> None:
        async with self.__async_engine.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # Service table
            await conn.execute(text("DROP TABLE IF EXISTS services CASCADE"))
            await conn.execute(
                text(
                    """
                    CREATE TABLE services(
                        id INT PRIMARY KEY,
                        category TEXT,
                        title TEXT,
                        description TEXT,
                        price INT,
                        embedding vector(768) NOT NULL
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    INSERT INTO services VALUES (:id, :category, :title, :description, :price, :embedding)
                    """
                ),
                [
                    {
                        "id": s.id,
                        "category": s.category,
                        "title": s.title,
                        "description": s.description,
                        "price": s.price,
                        "embedding": s.embedding,
                    }
                    for s in services
                ],
            )

            # Kursus table
            await conn.execute(text("DROP TABLE IF EXISTS kursus CASCADE"))
            await conn.execute(
                text(
                    """
                    CREATE TABLE kursus(
                        id INT PRIMARY KEY,
                        course_name TEXT,
                        level TEXT,
                        description TEXT,
                        price INT,
                        start_date TEXT,
                        end_date TEXT,
                        embedding vector(768) NOT NULL
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    INSERT INTO kursus VALUES (:id, :course_name, :level, :description, :price, :start_date, :end_date, :embedding)
                    """
                ),
                [
                    {
                        "id": k.id,
                        "course_name": k.course_name,
                        "level": k.level,
                        "description": k.description,
                        "price": k.price,
                        "start_date": k.start_date,
                        "end_date": k.end_date,
                        "embedding": k.embedding,
                    }
                    for k in kursus_list
                ],
            )

            # FAQ table
            await conn.execute(text("DROP TABLE IF EXISTS faqs CASCADE"))
            await conn.execute(
                text(
                    """
                    CREATE TABLE faqs(
                        id INT PRIMARY KEY,
                        category TEXT,
                        title TEXT,
                        description TEXT,
                        embedding vector(768) NOT NULL
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    INSERT INTO faqs VALUES (:id, :category, :title, :description, :embedding)
                    """
                ),
                [
                    {
                        "id": f.id,
                        "category": f.category,
                        "title": f.title,
                        "description": f.description,
                        "embedding": f.embedding,
                    }
                    for f in faqs
                ],
            )
            await conn.commit()

    async def export_data(
        self,
    ) -> tuple[
        list[models.Service],
        list[models.Kursus],
        list[models.FAQ],
    ]:
        async with self.__async_engine.connect() as conn:
            service_task = asyncio.create_task(
                conn.execute(text("""SELECT * FROM services ORDER BY id ASC"""))
            )
            kursus_task = asyncio.create_task(
                conn.execute(text("""SELECT * FROM kursus ORDER BY id ASC"""))
            )
            faq_task = asyncio.create_task(
                conn.execute(text("""SELECT * FROM faqs ORDER BY id ASC"""))
            )

            service_results = (await service_task).mappings().fetchall()
            kursus_results = (await kursus_task).mappings().fetchall()
            faq_results = (await faq_task).mappings().fetchall()

            services = [models.Service.model_validate(s) for s in service_results]
            kursus_list = [models.Kursus.model_validate(k) for k in kursus_results]
            faqs = [models.FAQ.model_validate(f) for f in faq_results]

            return services, kursus_list, faqs

    async def search_services(
        self, query_embedding: list[float], similarity_threshold: float, top_k: int
    ) -> tuple[list[Any], Optional[str]]:
        async with self.__async_engine.connect() as conn:
            sql = """
                SELECT id, category, title, description, price
                FROM services
                WHERE (embedding <=> :query_embedding) < :similarity_threshold
                ORDER BY (embedding <=> :query_embedding)
                LIMIT :top_k
                """
            s = text(sql)
            params = {
                "query_embedding": query_embedding,
                "similarity_threshold": similarity_threshold,
                "top_k": top_k,
            }
            results = (await conn.execute(s, params)).mappings().fetchall()
        res = [r for r in results]
        return res, format_sql(sql, params)

    async def search_kursus(
        self, query_embedding: list[float], similarity_threshold: float, top_k: int
    ) -> tuple[list[Any], Optional[str]]:
        async with self.__async_engine.connect() as conn:
            sql = """
                SELECT id, course_name, level, description, price, start_date, end_date
                FROM kursus
                WHERE (embedding <=> :query_embedding) < :similarity_threshold
                ORDER BY (embedding <=> :query_embedding)
                LIMIT :top_k
                """
            s = text(sql)
            params = {
                "query_embedding": query_embedding,
                "similarity_threshold": similarity_threshold,
                "top_k": top_k,
            }
            results = (await conn.execute(s, params)).mappings().fetchall()
        res = [r for r in results]
        return res, format_sql(sql, params)

    async def search_faqs(
        self, query_embedding: list[float], similarity_threshold: float, top_k: int
    ) -> tuple[list[Any], Optional[str]]:
        async with self.__async_engine.connect() as conn:
            sql = """
                SELECT id, category, title, description
                FROM faqs
                WHERE (embedding <=> :query_embedding) < :similarity_threshold
                ORDER BY (embedding <=> :query_embedding)
                LIMIT :top_k
                """
            s = text(sql)
            params = {
                "query_embedding": query_embedding,
                "similarity_threshold": similarity_threshold,
                "top_k": top_k,
            }
            results = (await conn.execute(s, params)).mappings().fetchall()
        res = [r for r in results]
        return res, format_sql(sql, params)

    async def close(self):
        await self.__async_engine.dispose()

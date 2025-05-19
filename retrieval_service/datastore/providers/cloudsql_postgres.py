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
from typing import Any, Literal, Optional

import asyncpg
from google.cloud.sql.connector import Connector, RefreshStrategy
from pgvector.asyncpg import register_vector
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

import models

from .. import datastore
from .postgres import Client as PostgresClient

CLOUD_SQL_PG_IDENTIFIER = "cloudsql-postgres"


class Config(BaseModel, datastore.AbstractConfig):
    kind: Literal["cloudsql-postgres"]
    project: str
    region: str
    instance: str
    user: str
    password: str
    database: str


class Client(datastore.Client[Config]):
    __pg_client: PostgresClient
    __connector: Optional[Connector] = None

    @datastore.classproperty
    def kind(cls):
        return CLOUD_SQL_PG_IDENTIFIER

    def __init__(self, async_engine: AsyncEngine):
        self.__pg_client = PostgresClient(async_engine)

    @classmethod
    async def create(cls, config: Config) -> "Client":
        async def getconn() -> asyncpg.Connection:
            if cls.__connector is None:
                loop = asyncio.get_running_loop()
                cls.__connector = Connector(
                    loop=loop, refresh_strategy=RefreshStrategy.LAZY
                )

            conn: asyncpg.Connection = await cls.__connector.connect_async(
                # Cloud SQL instance connection name
                f"{config.project}:{config.region}:{config.instance}",
                "asyncpg",
                user=f"{config.user}",
                password=f"{config.password}",
                db=f"{config.database}",
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
        await self.__pg_client.initialize_data(services, kursus_list, faqs)

    async def export_data(
        self,
    ) -> tuple[
        list[models.Service],
        list[models.Kursus],
        list[models.FAQ],
    ]:
        return await self.__pg_client.export_data()

    async def search_services(
        self, query_embedding: list[float], similarity_threshold: float, top_k: int
    ) -> tuple[list[Any], Optional[str]]:
        return await self.__pg_client.search_services(query_embedding, similarity_threshold, top_k)

    async def search_kursus(
        self, query_embedding: list[float], similarity_threshold: float, top_k: int
    ) -> tuple[list[Any], Optional[str]]:
        return await self.__pg_client.search_kursus(query_embedding, similarity_threshold, top_k)

    async def search_faqs(
        self, query_embedding: list[float], similarity_threshold: float, top_k: int
    ) -> tuple[list[Any], Optional[str]]:
        return await self.__pg_client.search_faqs(query_embedding, similarity_threshold, top_k)

    async def close(self):
        await self.__pg_client.close()

    # Contoh: definisi tabel baru
    CREATE_SERVICE_TABLE = """
    CREATE TABLE IF NOT EXISTS services (
        id SERIAL PRIMARY KEY,
        category TEXT,
        title TEXT,
        description TEXT,
        price INTEGER,
        embedding VECTOR(384)
    );
    """

    CREATE_KURSUS_TABLE = """
    CREATE TABLE IF NOT EXISTS kursus (
        id SERIAL PRIMARY KEY,
        course_name TEXT,
        level TEXT,
        description TEXT,
        price INTEGER,
        start_date DATE,
        end_date DATE,
        embedding VECTOR(384)
    );
    """

    CREATE_FAQ_TABLE = """
    CREATE TABLE IF NOT EXISTS faqs (
        id SERIAL PRIMARY KEY,
        category TEXT,
        title TEXT,
        description TEXT,
        embedding VECTOR(384)
    );
    """

    # Insert logic baru
    async def insert_services(self, services: list):
        # services: List[Service]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO services (id, category, title, description, price, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    category=EXCLUDED.category,
                    title=EXCLUDED.title,
                    description=EXCLUDED.description,
                    price=EXCLUDED.price,
                    embedding=EXCLUDED.embedding
                """,
                [
                    (
                        s.id,
                        s.category,
                        s.title,
                        s.description,
                        s.price,
                        s.embedding,
                    )
                    for s in services
                ],
            )

    async def insert_kursus(self, kursus_list: list):
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO kursus (id, course_name, level, description, price, start_date, end_date, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    course_name=EXCLUDED.course_name,
                    level=EXCLUDED.level,
                    description=EXCLUDED.description,
                    price=EXCLUDED.price,
                    start_date=EXCLUDED.start_date,
                    end_date=EXCLUDED.end_date,
                    embedding=EXCLUDED.embedding
                """,
                [
                    (
                        k.id,
                        k.course_name,
                        k.level,
                        k.description,
                        k.price,
                        k.start_date,
                        k.end_date,
                        k.embedding,
                    )
                    for k in kursus_list
                ],
            )

    async def insert_faqs(self, faqs: list):
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO faqs (id, category, title, description, embedding)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    category=EXCLUDED.category,
                    title=EXCLUDED.title,
                    description=EXCLUDED.description,
                    embedding=EXCLUDED.embedding
                """,
                [
                    (
                        f.id,
                        f.category,
                        f.title,
                        f.description,
                        f.embedding,
                    )
                    for f in faqs
                ],
            )

    # Search logic baru
    async def search_services(self, query_embedding, similarity_threshold, top_k):
        async with self.pool.acquire() as conn:
            sql = """
            SELECT id, category, title, description, price, embedding,
                1 - (embedding <=> $1::vector) AS similarity
            FROM services
            WHERE 1 - (embedding <=> $1::vector) >= $2
            ORDER BY similarity DESC
            LIMIT $3
            """
            rows = await conn.fetch(sql, query_embedding, similarity_threshold, top_k)
            return [dict(row) for row in rows], sql

    async def search_kursus(self, query_embedding, similarity_threshold, top_k):
        async with self.pool.acquire() as conn:
            sql = """
            SELECT id, course_name, level, description, price, start_date, end_date, embedding,
                1 - (embedding <=> $1::vector) AS similarity
            FROM kursus
            WHERE 1 - (embedding <=> $1::vector) >= $2
            ORDER BY similarity DESC
            LIMIT $3
            """
            rows = await conn.fetch(sql, query_embedding, similarity_threshold, top_k)
            return [dict(row) for row in rows], sql

    async def search_faqs(self, query_embedding, similarity_threshold, top_k):
        async with self.pool.acquire() as conn:
            sql = """
            SELECT id, category, title, description, embedding,
                1 - (embedding <=> $1::vector) AS similarity
            FROM faqs
            WHERE 1 - (embedding <=> $1::vector) >= $2
            ORDER BY similarity DESC
            LIMIT $3
            """
            rows = await conn.fetch(sql, query_embedding, similarity_threshold, top_k)
            return [dict(row) for row in rows], sql

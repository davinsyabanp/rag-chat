# Copyright 2024 Google LLC
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

from typing import Any, Literal, Optional

import asyncpg
from google.cloud.alloydb.connector import AsyncConnector, RefreshStrategy
from pgvector.asyncpg import register_vector
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

import models

from .. import datastore
from .postgres import Client as PostgresClient

ALLOYDB_PG_IDENTIFIER = "alloydb-postgres"


class Config(BaseModel, datastore.AbstractConfig):
    kind: Literal["alloydb-postgres"]
    project: str
    region: str
    cluster: str
    instance: str
    user: str
    password: str
    database: str


class Client(datastore.Client[Config]):
    __connector: Optional[AsyncConnector] = None
    __pg_client: PostgresClient

    @datastore.classproperty
    def kind(cls):
        return ALLOYDB_PG_IDENTIFIER

    def __init__(self, async_engine: AsyncEngine):
        self.__pg_client = PostgresClient(async_engine)

    @classmethod
    async def create(cls, config: Config) -> "Client":
        async def getconn() -> asyncpg.Connection:
            if cls.__connector is None:
                cls.__connector = AsyncConnector(refresh_strategy=RefreshStrategy.LAZY)

            conn: asyncpg.Connection = await cls.__connector.connect(
                # Alloydb instance connection name
                f"projects/{config.project}/locations/{config.region}/clusters/{config.cluster}/instances/{config.instance}",
                "asyncpg",
                user=f"{config.user}",
                password=f"{config.password}",
                db=f"{config.database}",
                ip_type="PUBLIC",
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
        kursus: list[models.Kursus],
        faqs: list[models.Faq],
    ) -> None:
        await self.__pg_client.initialize_data(services, kursus, faqs)

    async def export_data(
        self,
    ) -> tuple[
        list[models.Service],
        list[models.Kursus],
        list[models.Faq],
    ]:
        return await self.__pg_client.export_data()

    # --- Service ---
    async def get_service_by_id(
        self, id: int
    ) -> tuple[Optional[models.Service], Optional[str]]:
        return await self.__pg_client.get_service_by_id(id)

    async def get_service_by_category(
        self, category: str
    ) -> tuple[list[models.Service], Optional[str]]:
        return await self.__pg_client.get_service_by_category(category)

    async def services_search(
        self, query_embedding: list[float], similarity_threshold: float, top_k: int
    ) -> tuple[list[models.Service], Optional[str]]:
        return await self.__pg_client.services_search(
            query_embedding, similarity_threshold, top_k
        )

    # --- Course ---
    async def get_course_by_id(
        self, id: int
    ) -> tuple[Optional[models.Kursus], Optional[str]]:
        return await self.__pg_client.get_course_by_id(id)

    async def get_course_by_level(
        self, level: str
    ) -> tuple[list[models.Kursus], Optional[str]]:
        return await self.__pg_client.get_course_by_level(level)

    async def courses_search(
        self, query_embedding: list[float], similarity_threshold: float, top_k: int
    ) -> tuple[list[models.Kursus], Optional[str]]:
        return await self.__pg_client.courses_search(
            query_embedding, similarity_threshold, top_k
        )

    # --- FAQ ---
    async def get_faq_by_id(
        self, id: int
    ) -> tuple[Optional[models.Faq], Optional[str]]:
        return await self.__pg_client.get_faq_by_id(id)

    async def get_faq_by_category(
        self, category: str
    ) -> tuple[list[models.Faq], Optional[str]]:
        return await self.__pg_client.get_faq_by_category(category)

    async def faqs_search(
        self, query_embedding: list[float], similarity_threshold: float, top_k: int
    ) -> tuple[list[models.Faq], Optional[str]]:
        return await self.__pg_client.faqs_search(
            query_embedding, similarity_threshold, top_k
        )

    async def close(self):
        await self.__pg_client.close()

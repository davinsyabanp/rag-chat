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
from typing import Any, Literal, Optional

import pymysql
from google.cloud.sql.connector import Connector, RefreshStrategy
from pydantic import BaseModel
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine.base import Engine

import models

from .. import datastore

MYSQL_IDENTIFIER = "cloudsql-mysql"


class Config(BaseModel, datastore.AbstractConfig):
    kind: Literal["cloudsql-mysql"]
    project: str
    region: str
    instance: str
    user: str
    password: str
    database: str


class Client(datastore.Client[Config]):
    __pool: Engine
    __db_name: str
    __connector: Optional[Connector] = None

    @datastore.classproperty
    def kind(cls):
        return MYSQL_IDENTIFIER

    def __init__(self, pool: Engine, db_name: str):
        self.__pool = pool
        self.__db_name = db_name

    @classmethod
    def create_sync(cls, config: Config) -> "Client":
        def getconn() -> pymysql.Connection:
            if cls.__connector is None:
                cls.__connector = Connector(refresh_strategy=RefreshStrategy.LAZY)

            return cls.__connector.connect(
                # Cloud SQL instance connection name
                f"{config.project}:{config.region}:{config.instance}",
                "pymysql",
                user=f"{config.user}",
                password=f"{config.password}",
                db=f"{config.database}",
                autocommit=True,
            )

        pool = create_engine(
            "mysql+pymysql://",
            creator=getconn,
        )
        if pool is None:
            raise TypeError("pool not instantiated")
        return cls(pool, config.database)

    @classmethod
    async def create(cls, config: Config) -> "Client":
        loop = asyncio.get_running_loop()

        pool = await loop.run_in_executor(None, cls.create_sync, config)
        return pool

    def initialize_data_sync(
        self,
        services: list[models.Service],
        kursus_list: list[models.Kursus],
        faqs: list[models.Faq],
    ) -> None:
        with self.__pool.connect() as conn:
            # Drop and create service table
            conn.execute(text("DROP TABLE IF EXISTS services"))
            conn.execute(
                text(
                    """
                    CREATE TABLE services(
                      id INT PRIMARY KEY,
                      category TEXT,
                      title TEXT,
                      description TEXT,
                      price INT,
                      embedding vector(768) USING VARBINARY NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """INSERT INTO services VALUES (:id, :category, :title, :description, :price, string_to_vector(:embedding))"""
                ),
                parameters=[
                    {
                        "id": s.id,
                        "category": s.category,
                        "title": s.title,
                        "description": s.description,
                        "price": s.price,
                        "embedding": f"{s.embedding}",
                    }
                    for s in services
                ],
            )

            # Drop and create kursus table
            conn.execute(text("DROP TABLE IF EXISTS kursus"))
            conn.execute(
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
                      embedding vector(768) USING VARBINARY NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """INSERT INTO kursus VALUES (:id, :course_name, :level, :description, :price, :start_date, :end_date, string_to_vector(:embedding))"""
                ),
                parameters=[
                    {
                        "id": k.id,
                        "course_name": k.course_name,
                        "level": k.level,
                        "description": k.description,
                        "price": k.price,
                        "start_date": k.start_date,
                        "end_date": k.end_date,
                        "embedding": f"{k.embedding}",
                    }
                    for k in kursus_list
                ],
            )

            # Drop and create faq table
            conn.execute(text("DROP TABLE IF EXISTS faqs"))
            conn.execute(
                text(
                    """
                    CREATE TABLE faqs(
                      id INT PRIMARY KEY,
                      category TEXT,
                      title TEXT,
                      description TEXT,
                      embedding vector(768) USING VARBINARY NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """INSERT INTO faqs VALUES (:id, :category, :title, :description, string_to_vector(:embedding))"""
                ),
                parameters=[
                    {
                        "id": f.id,
                        "category": f.category,
                        "title": f.title,
                        "description": f.description,
                        "embedding": f"{f.embedding}",
                    }
                    for f in faqs
                ],
            )

    async def initialize_data(
        self,
        services: list[models.Service],
        kursus_list: list[models.Kursus],
        faqs: list[models.Faq],
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self.initialize_data_sync, services, kursus_list, faqs
        )

    def export_data_sync(
        self,
    ) -> tuple[
        list[models.Service],
        list[models.Kursus],
        list[models.Faq],
    ]:
        with self.__pool.connect() as conn:
            service_task = conn.execute(
                text(
                    """SELECT id, category, title, description, price, vector_to_string(embedding) as embedding FROM services ORDER BY id ASC"""
                )
            )
            kursus_task = conn.execute(
                text(
                    """SELECT id, course_name, level, description, price, start_date, end_date, vector_to_string(embedding) as embedding FROM kursus ORDER BY id ASC"""
                )
            )
            faq_task = conn.execute(
                text(
                    """SELECT id, category, title, description, vector_to_string(embedding) as embedding FROM faqs ORDER BY id ASC"""
                )
            )

            service_results = (service_task).mappings().fetchall()
            kursus_results = (kursus_task).mappings().fetchall()
            faq_results = (faq_task).mappings().fetchall()

            services = [models.Service.model_validate(s) for s in service_results]
            kursus_list = [models.Kursus.model_validate(k) for k in kursus_results]
            faqs = [models.Faq.model_validate(f) for f in faq_results]

            return services, kursus_list, faqs

    async def export_data(
        self,
    ) -> tuple[
        list[models.Service],
        list[models.Kursus],
        list[models.Faq],
    ]:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, self.export_data_sync)
        return res

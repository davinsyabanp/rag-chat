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

import datetime
from typing import Any, Literal, Optional

from google.cloud import spanner  # type: ignore
from google.cloud.spanner_v1 import JsonObject, param_types
from google.cloud.spanner_v1.database import Database
from google.cloud.spanner_v1.instance import Instance
from google.oauth2 import service_account  # type: ignore
from pydantic import BaseModel

import models

from .. import datastore

# Identifier for Spanner
SPANNER_IDENTIFIER = "spanner-postgres"


# Configuration model for Spanner
class Config(BaseModel, datastore.AbstractConfig):
    """
    Configuration model for Spanner.

    Attributes:
        kind (Literal["spanner"]): Type of datastore.
        project (str): Google Cloud project ID.
        instance (str): ID of the Spanner instance.
        database (str): ID of the Spanner database.
        service_account_key_file (str): Service Account Key File.
    """

    kind: Literal["spanner-postgres"]
    project: str
    instance: str
    database: str
    service_account_key_file: Optional[str] = None


# Client class for interacting with Spanner
class Client(datastore.Client[Config]):
    OPERATION_TIMEOUT_SECONDS = 240
    BATCH_SIZE = 1000
    SERVICE_COLUMNS = ["id", "category", "title", "description", "price", "embedding"]
    KURSUS_COLUMNS = [
        "id",
        "course_name",
        "level",
        "description",
        "price",
        "start_date",
        "end_date",
        "embedding",
    ]
    FAQ_COLUMNS = ["id", "category", "title", "description", "embedding"]

    @datastore.classproperty
    def kind(cls):
        return SPANNER_IDENTIFIER

    def __init__(self, client: spanner.Client, instance_id: str, database_id: str):
        """
        Initialize the Spanner client.

        Args:
            client (spanner.Client): Spanner client instance.
            instance_id (str): ID of the Spanner instance.
            database_id (str): ID of the Spanner database.
        """
        self.__client = client
        self.__instance_id = instance_id
        self.__database_id = database_id

        self.__instance = self.__client.instance(self.__instance_id)
        self.__database = self.__instance.database(self.__database_id)

    @classmethod
    async def create(cls, config: Config) -> "Client":
        """
        Create a Spanner client.

        Args:
            config (Config): Configuration for creating the client.

        Returns:
            Client: Initialized Spanner client.
        """
        client: spanner.Client

        if config.service_account_key_file is not None:
            credentials = service_account.Credentials.from_service_account_file(
                config.service_account_key_file
            )
            client = spanner.Client(project=config.project, credentials=credentials)
        else:
            client = spanner.Client(project=config.project)

        instance_id = config.instance
        instance = client.instance(instance_id)

        if not instance.exists():
            raise Exception(f"Instance with id: {instance_id} doesn't exist.")

        database_id = config.database
        database = instance.database(database_id)

        if not database.exists():
            raise Exception(f"Database with id: {database_id} doesn't exist.")

        return cls(client, instance_id, database_id)

    async def initialize_data(
        self,
        services: list[models.Service],
        kursus_list: list[models.Kursus],
        faqs: list[models.Faq],
    ) -> None:
        """
        Initialize data in the Spanner database by creating tables and inserting records.

        Args:
            services (list[models.Service]): list of services to be initialized.
            kursus_list (list[models.Kursus]): list of kursus to be initialized.
            faqs (list[models.Faq]): list of faqs to be initialized.
        Returns:
            None
        """
        # Initialize a list to store Data Definition Language (DDL) statements
        ddl = []

        # Create DDL statement to drop the 'services' table if it exists
        ddl.append("DROP TABLE IF EXISTS services")

        # Create DDL statement to create the 'services' table
        ddl.append(
            """
            CREATE TABLE services(
                id BIGINT PRIMARY KEY,
                category VARCHAR,
                title VARCHAR,
                description VARCHAR,
                price BIGINT,
                embedding FLOAT8[] NOT NULL
            )
            """
        )

        # Create DDL statement to drop the 'kursus' table if it exists
        ddl.append("DROP TABLE IF EXISTS kursus")

        # Create DDL statement to create the 'kursus' table
        ddl.append(
            """
            CREATE TABLE kursus(
                id BIGINT PRIMARY KEY,
                course_name VARCHAR,
                level VARCHAR,
                description VARCHAR,
                price BIGINT,
                start_date VARCHAR,
                end_date VARCHAR,
                embedding FLOAT8[] NOT NULL
            )
            """
        )

        # Create DDL statement to drop the 'faqs' table if it exists
        ddl.append("DROP TABLE IF EXISTS faqs")

        # Create DDL statement to create the 'faqs' table
        ddl.append(
            """
            CREATE TABLE faqs(
                id BIGINT PRIMARY KEY,
                category VARCHAR,
                title VARCHAR,
                description VARCHAR,
                embedding FLOAT8[] NOT NULL
            )
            """
        )

        # Update the schema using DDL statements
        operation = self.__database.update_ddl(ddl)

        print("Waiting for schema update operation to complete...")
        operation.result(self.OPERATION_TIMEOUT_SECONDS)
        print("Schema update operation completed")

        # Insert services
        values = [
            (
                s.id,
                s.category,
                s.title,
                s.description,
                s.price,
                s.embedding,
            )
            for s in services
        ]
        for i in range(0, len(values), self.BATCH_SIZE):
            records = values[i : i + self.BATCH_SIZE]
            with self.__database.batch() as batch:
                batch.insert(
                    table="services",
                    columns=self.SERVICE_COLUMNS,
                    values=records,
                )

        # Insert kursus
        values = [
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
        ]
        for i in range(0, len(values), self.BATCH_SIZE):
            records = values[i : i + self.BATCH_SIZE]
            with self.__database.batch() as batch:
                batch.insert(
                    table="kursus",
                    columns=self.KURSUS_COLUMNS,
                    values=records,
                )

        # Insert faqs
        values = [
            (
                f.id,
                f.category,
                f.title,
                f.description,
                f.embedding,
            )
            for f in faqs
        ]
        for i in range(0, len(values), self.BATCH_SIZE):
            records = values[i : i + self.BATCH_SIZE]
            with self.__database.batch() as batch:
                batch.insert(
                    table="faqs",
                    columns=self.FAQ_COLUMNS,
                    values=records,
                )

    async def export_data(
        self,
    ) -> tuple[
        list[models.Service],
        list[models.Kursus],
        list[models.Faq],
    ]:
        """
        Export data from the Spanner database.

        Returns:
            tuple: A tuple containing lists of services, kursus, and faqs.
        """
        services: list = []
        kursus_list: list = []
        faqs: list = []

        try:
            with self.__database.snapshot() as snapshot:
                service_results = snapshot.execute_sql(
                    "SELECT {} FROM services ORDER BY id ASC".format(
                        ",".join(self.SERVICE_COLUMNS)
                    )
                )
        except Exception as e:
            print(f"Error occurred while fetch services: {e}")
            return services, kursus_list, faqs

        services = [
            models.Service.model_validate(
                {key: value for key, value in zip(self.SERVICE_COLUMNS, a)}
            )
            for a in service_results
        ]

        try:
            with self.__database.snapshot() as snapshot:
                kursus_results = snapshot.execute_sql(
                    "SELECT {} FROM kursus ORDER BY id ASC".format(
                        ",".join(self.KURSUS_COLUMNS)
                    )
                )
        except Exception as e:
            print(f"Error occurred while fetch kursus: {e}")
            return services, kursus_list, faqs

        kursus_list = [
            models.Kursus.model_validate(
                {key: value for key, value in zip(self.KURSUS_COLUMNS, a)}
            )
            for a in kursus_results
        ]

        try:
            with self.__database.snapshot() as snapshot:
                faq_results = snapshot.execute_sql(
                    "SELECT {} FROM faqs ORDER BY id ASC".format(
                        ",".join(self.FAQ_COLUMNS)
                    )
                )
        except Exception as e:
            print(f"Error occurred while fetch faqs: {e}")
            return services, kursus_list, faqs

        faqs = [
            models.Faq.model_validate(
                {key: value for key, value in zip(self.FAQ_COLUMNS, a)}
            )
            for a in faq_results
        ]

        return services, kursus_list, faqs

    async def search_services(self, query_embedding: list[float], similarity_threshold: float, top_k: int):
        with self.__database.snapshot() as snapshot:
            query = """
                SELECT id, category, title, description, price, embedding
                FROM (
                    SELECT id, category, title, description, price, embedding,
                       spanner.cosine_distance(embedding, $1) AS similarity
                    FROM services
                ) AS sorted_services
                WHERE (1 - similarity) > $2
                ORDER BY similarity
                LIMIT $3
            """
            results = snapshot.execute_sql(
                sql=query,
                params={
                    "p1": query_embedding,
                    "p2": similarity_threshold,
                    "p3": top_k,
                },
                param_types={
                    "p1": param_types.Array(param_types.FLOAT64),
                    "p2": param_types.FLOAT64,
                    "p3": param_types.INT64,
                },
            )
        return [
            models.Service.model_validate(
                {key: value for key, value in zip(self.SERVICE_COLUMNS, a)}
            )
            for a in results
        ], query

    async def search_kursus(self, query_embedding: list[float], similarity_threshold: float, top_k: int):
        with self.__database.snapshot() as snapshot:
            query = """
                SELECT id, course_name, level, description, price, start_date, end_date, embedding
                FROM (
                    SELECT id, course_name, level, description, price, start_date, end_date, embedding,
                       spanner.cosine_distance(embedding, $1) AS similarity
                    FROM kursus
                ) AS sorted_kursus
                WHERE (1 - similarity) > $2
                ORDER BY similarity
                LIMIT $3
            """
            results = snapshot.execute_sql(
                sql=query,
                params={
                    "p1": query_embedding,
                    "p2": similarity_threshold,
                    "p3": top_k,
                },
                param_types={
                    "p1": param_types.Array(param_types.FLOAT64),
                    "p2": param_types.FLOAT64,
                    "p3": param_types.INT64,
                },
            )
        return [
            models.Kursus.model_validate(
                {key: value for key, value in zip(self.KURSUS_COLUMNS, a)}
            )
            for a in results
        ], query

    async def search_faqs(self, query_embedding: list[float], similarity_threshold: float, top_k: int):
        with self.__database.snapshot() as snapshot:
            query = """
                SELECT id, category, title, description, embedding
                FROM (
                    SELECT id, category, title, description, embedding,
                       spanner.cosine_distance(embedding, $1) AS similarity
                    FROM faqs
                ) AS sorted_faqs
                WHERE (1 - similarity) > $2
                ORDER BY similarity
                LIMIT $3
            """
            results = snapshot.execute_sql(
                sql=query,
                params={
                    "p1": query_embedding,
                    "p2": similarity_threshold,
                    "p3": top_k,
                },
                param_types={
                    "p1": param_types.Array(param_types.FLOAT64),
                    "p2": param_types.FLOAT64,
                    "p3": param_types.INT64,
                },
            )
        return [
            models.Faq.model_validate(
                {key: value for key, value in zip(self.FAQ_COLUMNS, a)}
            )
            for a in results
        ], query

    async def close(self):
        """
        Closes the database client connection.
        """
        self.__client.close()

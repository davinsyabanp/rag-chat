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
from datetime import datetime, timedelta
from typing import Any, Literal, Optional

from google.cloud.firestore import AsyncClient  # type: ignore
from google.cloud.firestore_v1.async_collection import AsyncCollectionReference
from google.cloud.firestore_v1.async_query import AsyncQuery
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from pydantic import BaseModel

import models

from .. import datastore

FIRESTORE_IDENTIFIER = "firestore"


class Config(BaseModel, datastore.AbstractConfig):
    kind: Literal["firestore"]
    projectId: Optional[str]


class Client(datastore.Client[Config]):
    __client: AsyncClient

    @datastore.classproperty
    def kind(cls):
        return FIRESTORE_IDENTIFIER

    def __init__(self, client: AsyncClient):
        self.__client = client
        self.__service_collection = AsyncQuery(self.__client.collection("services"))
        self.__kursus_collection = AsyncQuery(self.__client.collection("kursus"))
        self.__faq_collection = AsyncQuery(self.__client.collection("faqs"))

    @classmethod
    async def create(cls, config: Config) -> "Client":
        return cls(AsyncClient(project=config.projectId))

    async def __delete_collections(
        self, collection_list: list[AsyncCollectionReference]
    ):
        # Checks if collection exists and deletes all documents
        delete_tasks = []
        for collection_ref in collection_list:
            collection_exists = collection_ref.limit(1).stream()
            if not collection_exists:
                continue

            docs = collection_ref.stream()
            async for doc in docs:
                delete_tasks.append(asyncio.create_task(doc.reference.delete()))
        await asyncio.gather(*delete_tasks)

    async def parse_index_info(self, line: str) -> tuple[str, str]:
        # Extract collection and index-id from file path
        parts = line.split("/")
        collection_name = parts[-3]
        index_id = parts[-1]
        return collection_name, index_id

    async def __get_indices(self) -> dict[str, str]:
        list_vector_index_process = await asyncio.create_subprocess_exec(
            "gcloud",
            "alpha",
            "firestore",
            "indexes",
            "composite",
            "list",
            "--database=(default)",
            "--format=value(name)",  # prints name field
            stdout=asyncio.subprocess.PIPE,
        )

        # Capture output and ignore stderr
        stdout, __ = await list_vector_index_process.communicate()

        # Decode and format output
        index_lines = stdout.decode().strip().split("\n")

        indices = {}

        # Create a dict with collections and their corresponding vector index.
        for line in index_lines:
            if line:
                collection, index_id = await self.parse_index_info(line)
                indices[collection] = index_id

        return indices

    async def __delete_vector_index(self, indices: list[str]):
        # Check if the collection exists and deletes all indexes
        for index in indices:
            if index:
                delete_vector_index = await asyncio.create_subprocess_exec(
                    "gcloud",
                    "alpha",
                    "firestore",
                    "indexes",
                    "composite",
                    "delete",
                    index,
                    "--database=(default)",
                    "--quiet",  # Added to suppress delete warning
                )
                await delete_vector_index.wait()

    async def __create_vector_index(self, collection_name: str):
        create_vector_index = await asyncio.create_subprocess_exec(
            "gcloud",
            "alpha",
            "firestore",
            "indexes",
            "composite",
            "create",
            f"--collection-group={collection_name}",
            "--query-scope=COLLECTION",
            '--field-config=field-path=embedding,vector-config={"dimension":768,"flat":"{}"}',
            "--database=(default)",
        )
        await create_vector_index.wait()

    async def initialize_data(
        self,
        services: list[models.Service],
        kursus_list: list[models.Kursus],
        faqs: list[models.Faq],
    ) -> None:
        # Delete collections if exist
        service_ref = self.__client.collection("services")
        kursus_ref = self.__client.collection("kursus")
        faq_ref = self.__client.collection("faqs")
        await self.__delete_collections([service_ref, kursus_ref, faq_ref])

        # Insert Service
        create_service_tasks = []
        for service in services:
            create_service_tasks.append(
                self.__client.collection("services")
                .document(str(service.id))
                .set(
                    {
                        "category": service.category,
                        "title": service.title,
                        "description": service.description,
                        "price": service.price,
                        "embedding": Vector(service.embedding or []),
                    }
                )
            )
        await asyncio.gather(*create_service_tasks)

        # Insert Kursus
        create_kursus_tasks = []
        for kursus in kursus_list:
            create_kursus_tasks.append(
                self.__client.collection("kursus")
                .document(str(kursus.id))
                .set(
                    {
                        "course_name": kursus.course_name,
                        "level": kursus.level,
                        "description": kursus.description,
                        "price": kursus.price,
                        "start_date": kursus.start_date,
                        "end_date": kursus.end_date,
                        "embedding": Vector(kursus.embedding or []),
                    }
                )
            )
        await asyncio.gather(*create_kursus_tasks)

        # Insert Faq
        create_faq_tasks = []
        for faq in faqs:
            create_faq_tasks.append(
                self.__client.collection("faqs")
                .document(str(faq.id))
                .set(
                    {
                        "category": faq.category,
                        "title": faq.title,
                        "description": faq.description,
                        "embedding": Vector(faq.embedding or []),
                    }
                )
            )
        await asyncio.gather(*create_faq_tasks)

        # Initialize single-field vector indexes
        await self.__create_vector_index("services")
        await self.__create_vector_index("kursus")
        await self.__create_vector_index("faqs")

    async def export_data(
        self,
    ) -> tuple[
        list[models.Service],
        list[models.Kursus],
        list[models.Faq],
    ]:
        service_docs = self.__client.collection("services").stream()
        kursus_docs = self.__client.collection("kursus").stream()
        faq_docs = self.__client.collection("faqs").stream()

        services = []
        async for doc in service_docs:
            d = doc.to_dict()
            d["id"] = int(doc.id)
            d["embedding"] = list(d.get("embedding", []))
            services.append(models.Service.model_validate(d))

        kursus_list = []
        async for doc in kursus_docs:
            d = doc.to_dict()
            d["id"] = int(doc.id)
            d["embedding"] = list(d.get("embedding", []))
            kursus_list.append(models.Kursus.model_validate(d))

        faqs = []
        async for doc in faq_docs:
            d = doc.to_dict()
            d["id"] = int(doc.id)
            d["embedding"] = list(d.get("embedding", []))
            faqs.append(models.Faq.model_validate(d))

        return services, kursus_list, faqs

    async def get_service_by_id(self, id: int) -> models.Service:
        doc = await self.__client.collection("services").document(str(id)).get()
        d = doc.to_dict()
        d["id"] = int(doc.id)
        d["embedding"] = list(d.get("embedding", []))
        return models.Service.model_validate(d)

    async def get_kursus_by_id(self, id: int) -> models.Kursus:
        doc = await self.__client.collection("kursus").document(str(id)).get()
        d = doc.to_dict()
        d["id"] = int(doc.id)
        d["embedding"] = list(d.get("embedding", []))
        return models.Kursus.model_validate(d)

    async def get_faq_by_id(self, id: int) -> models.Faq:
        doc = await self.__client.collection("faqs").document(str(id)).get()
        d = doc.to_dict()
        d["id"] = int(doc.id)
        d["embedding"] = list(d.get("embedding", []))
        return models.Faq.model_validate(d)

    async def search_services(self, query_embedding: list[float], similarity_threshold: float, top_k: int):
        query = self.__service_collection.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_embedding),
            distance_measure=DistanceMeasure.DOT_PRODUCT,
            limit=top_k,
        )
        docs = query.stream()
        results = []
        async for doc in docs:
            d = doc.to_dict()
            d["id"] = int(doc.id)
            d["embedding"] = list(d.get("embedding", []))
            results.append(models.Service.model_validate(d))
        return results

    async def search_kursus(self, query_embedding: list[float], similarity_threshold: float, top_k: int):
        query = self.__kursus_collection.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_embedding),
            distance_measure=DistanceMeasure.DOT_PRODUCT,
            limit=top_k,
        )
        docs = query.stream()
        results = []
        async for doc in docs:
            d = doc.to_dict()
            d["id"] = int(doc.id)
            d["embedding"] = list(d.get("embedding", []))
            results.append(models.Kursus.model_validate(d))
        return results

    async def search_faqs(self, query_embedding: list[float], similarity_threshold: float, top_k: int):
        query = self.__faq_collection.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_embedding),
            distance_measure=DistanceMeasure.DOT_PRODUCT,
            limit=top_k,
        )
        docs = query.stream()
        results = []
        async for doc in docs:
            d = doc.to_dict()
            d["id"] = int(doc.id)
            d["embedding"] = list(d.get("embedding", []))
            results.append(models.Faq.model_validate(d))
        return results

    async def close(self):
        self.__client.close()

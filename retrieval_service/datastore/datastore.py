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

import csv
from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, TypeVar

from .models import Faq, Kursus, Service

import models


class AbstractConfig(ABC):
    kind: str


C = TypeVar("C", bound=AbstractConfig)


class classproperty:
    def __init__(self, func):
        self.fget = func

    def __get__(self, _, owner):
        return self.fget(owner)


class Client(ABC, Generic[C]):
    @classproperty
    @abstractmethod
    def kind(cls):
        pass

    @classmethod
    @abstractmethod
    async def create(cls, config: C) -> "Client":
        pass

    async def load_dataset(
        self, services_path, kursus_path, faq_path
    ) -> tuple[List[Service], List[Kursus], List[Faq]]:
        services: List[Service] = []
        with open(services_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=",")
            services = [Service.model_validate(line) for line in reader]

        kursus_list: List[Kursus] = []
        with open(kursus_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=",")
            kursus_list = [Kursus.model_validate(line) for line in reader]

        faqs: List[Faq] = []
        with open(faq_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=",")
            faqs = [Faq.model_validate(line) for line in reader]
        return services, kursus_list, faqs

    async def export_dataset(
        self,
        services,
        kursus_list,
        faqs,
        services_new_path,
        kursus_new_path,
        faqs_new_path,
    ) -> None:
        with open(services_new_path, "w") as f:
            col_names = [
                "id",
                "name",
                "description",
                "category",
                "hour",
                "content",
                "embedding",
            ]
            writer = csv.DictWriter(f, col_names, delimiter=",")
            writer.writeheader()
            for a in services:
                writer.writerow(a.model_dump())

        with open(kursus_new_path, "w") as f:
            col_names = [
                "id",
                "name",
                "description",
                "duration",
                "price",
                "content",
                "embedding",
            ]
            writer = csv.DictWriter(f, col_names, delimiter=",")
            writer.writeheader()
            for a in kursus_list:
                writer.writerow(a.model_dump())

        with open(faqs_new_path, "w") as f:
            col_names = ["id", "question", "answer", "embedding"]
            writer = csv.DictWriter(f, col_names, delimiter=",")
            writer.writeheader()
            for a in faqs:
                writer.writerow(a.model_dump())

    @abstractmethod
    async def initialize_data(
        self,
        services: list[Service],
        kursus_list: list[Kursus],
        faqs: list[Faq],
    ) -> None:
        pass

    @abstractmethod
    async def export_data(
        self,
    ) -> tuple[list[Service], list[Kursus], list[Faq]]:
        pass

    @abstractmethod
    async def get_service_by_id(
        self, id: int
    ) -> tuple[Optional[Service], Optional[str]]:
        raise NotImplementedError("Subclass should implement this!")

    @abstractmethod
    async def search_services(
        self,
        name: Optional[str] = None,
        category: Optional[str] = None,
    ) -> tuple[list[Service], Optional[str]]:
        raise NotImplementedError("Subclass should implement this!")

    @abstractmethod
    async def get_kursus_by_id(self, id: int) -> tuple[Optional[Kursus], Optional[str]]:
        raise NotImplementedError("Subclass should implement this!")

    @abstractmethod
    async def search_kursus(
        self,
        name: Optional[str] = None,
        category: Optional[str] = None,
    ) -> tuple[list[Kursus], Optional[str]]:
        raise NotImplementedError("Subclass should implement this!")

    @abstractmethod
    async def get_faq_by_id(self, id: int) -> tuple[Optional[Faq], Optional[str]]:
        raise NotImplementedError("Subclass should implement this!")

    @abstractmethod
    async def search_faqs(
        self, query_embedding: list[float], similarity_threshold: float, top_k: int
    ) -> tuple[list[Any], Optional[str]]:
        raise NotImplementedError("Subclass should implement this!")

    @abstractmethod
    async def close(self):
        pass


async def create(config: AbstractConfig) -> Client:
    for cls in Client.__subclasses__():
        if config.kind == cls.kind:
            client = await cls.create(config)  # type: ignore

            return client
    raise TypeError(f"No clients of kind '{config.kind}'")

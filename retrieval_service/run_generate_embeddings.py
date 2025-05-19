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
import csv

from langchain_google_vertexai import VertexAIEmbeddings

import models
from app import EMBEDDING_MODEL_NAME


async def main() -> None:
    embed_service = VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Service embeddings
    services: list[models.Service] = []
    with open("../data/service_dummy.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for line in reader:
            service = models.Service.model_validate(line)
            # Gabungkan title + description untuk embedding
            content = f"{service.title}. {service.description}"
            service.embedding = embed_service.embed_query(content)
            services.append(service)

    # Kursus embeddings
    kursus_list: list[models.Kursus] = []
    with open("../data/kursus_dummy.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for line in reader:
            kursus = models.Kursus.model_validate(line)
            content = f"{kursus.course_name}. {kursus.description}"
            kursus.embedding = embed_service.embed_query(content)
            kursus_list.append(kursus)

    # FAQ embeddings
    faqs: list[models.FAQ] = []
    with open("../data/faq_dummy.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for line in reader:
            faq = models.FAQ.model_validate(line)
            content = f"{faq.title}. {faq.description}"
            faq.embedding = embed_service.embed_query(content)
            faqs.append(faq)

    print("Completed embedding generation.")

    # Write service embeddings
    with open("../data/service_dummy.csv.new", "w", encoding="utf-8", newline="") as f:
        col_names = [
            "id",
            "category",
            "title",
            "description",
            "price",
            "embedding",
        ]
        writer = csv.DictWriter(f, col_names, delimiter=",")
        writer.writeheader()
        for service in services:
            writer.writerow(service.model_dump())

    # Write kursus embeddings
    with open("../data/kursus_dummy.csv.new", "w", encoding="utf-8", newline="") as f:
        col_names = [
            "id",
            "course_name",
            "level",
            "description",
            "price",
            "start_date",
            "end_date",
            "embedding",
        ]
        writer = csv.DictWriter(f, col_names, delimiter=",")
        writer.writeheader()
        for kursus in kursus_list:
            writer.writerow(kursus.model_dump())

    # Write faq embeddings
    with open("../data/faq_dummy.csv.new", "w", encoding="utf-8", newline="") as f:
        col_names = [
            "id",
            "category",
            "title",
            "description",
            "embedding",
        ]
        writer = csv.DictWriter(f, col_names, delimiter=",")
        writer.writeheader()
        for faq in faqs:
            writer.writerow(faq.model_dump())

    print("Wrote data to CSV.")


if __name__ == "__main__":
    asyncio.run(main())

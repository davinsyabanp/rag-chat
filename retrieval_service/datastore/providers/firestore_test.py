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

from datetime import datetime
from typing import Dict

from google.cloud.firestore import AsyncClient, Client  # type: ignore
from google.cloud.firestore_v1.base_query import FieldFilter

import models

from . import firestore as firestore_provider


class MockDocument(Dict):
    """
    Mock firestore document.
    """

    id: int
    content: Dict

    def __init__(self, id, content):
        self.id = id
        self.content = content

    def to_dict(self):
        return self.content


class MockCollection(Dict):
    """
    Mock firestore collection.
    """

    collection_name: str
    documents: Dict[int, MockDocument]

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.documents = {}

    def where(self, *args, **kwargs):
        return self.documents.values()

    def select(self, *args):
        return self.documents.values()


class MockFirestoreClient(AsyncClient):
    """
    Mock firestore client.
    """

    collections: Dict[str, MockCollection]

    def __init__(self):
        self.collections = {}

    def collection(self, collection_name: str):
        return self.collections[collection_name]


async def mock_client(mock_firestore_client: MockFirestoreClient):
    return firestore_provider.Client(mock_firestore_client)


@pytest.mark.asyncio
async def test_firestore_service_kursus_faq():
    # Dummy Service
    service_doc = MockDocument(
        1,
        {
            "id": 1,
            "category": "Terjemahan",
            "title": "Terjemah Ijazah",
            "description": "Penerjemahan ijazah dari/ke Bahasa Indonesia, Arab, atau Inggris.",
            "price": 75000,
            "embedding": [0.1] * 768,
        },
    )
    service_collection = MockCollection("services")
    service_collection.documents[1] = service_doc

    # Dummy Kursus
    kursus_doc = MockDocument(
        1,
        {
            "id": 1,
            "course_name": "Kursus Bahasa Inggris Dasar",
            "level": "Dasar",
            "description": "Pengenalan grammar dasar, vocabulary umum, dan percakapan sehari-hari.",
            "price": 500000,
            "start_date": "6/1/2025",
            "end_date": "7/15/2025",
            "embedding": [0.3] * 768,
        },
    )
    kursus_collection = MockCollection("kursus")
    kursus_collection.documents[1] = kursus_doc

    # Dummy Faq
    faq_doc = MockDocument(
        1,
        {
            "id": 1,
            "category": "Pendaftaran",
            "title": "Bagaimana cara mendaftar kursus?",
            "description": "Kunjungi https://pusatbahasa.uinjkt.ac.id/pendaftaran, isi formulir online, dan lakukan pembayaran.",
            "embedding": [0.4] * 768,
        },
    )
    faq_collection = MockCollection("faqs")
    faq_collection.documents[1] = faq_doc

    mock_firestore_client = MockFirestoreClient()
    mock_firestore_client.collections["services"] = service_collection
    mock_firestore_client.collections["kursus"] = kursus_collection
    mock_firestore_client.collections["faqs"] = faq_collection

    client = await mock_client(mock_firestore_client)

    # Test get_service_by_id
    service = await client.get_service_by_id(1)
    assert service.id == 1
    assert service.title == "Terjemah Ijazah"

    # Test get_kursus_by_id
    kursus = await client.get_kursus_by_id(1)
    assert kursus.id == 1
    assert kursus.course_name == "Kursus Bahasa Inggris Dasar"

    # Test get_faq_by_id
    faq = await client.get_faq_by_id(1)
    assert faq.id == 1
    assert faq.title == "Bagaimana cara mendaftar kursus?"

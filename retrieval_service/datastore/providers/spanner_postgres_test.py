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

from datetime import datetime
from ipaddress import IPv4Address
from typing import Any, AsyncGenerator, Generator, List, Optional

import pytest
import pytest_asyncio
from csv_diff import compare, load_csv  # type: ignore
from google.cloud import spanner  # type: ignore
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect
from google.cloud.spanner_v1 import JsonObject, param_types
from google.cloud.spanner_v1.database import Database
from google.cloud.spanner_v1.instance import Instance

import models

from .. import datastore
from . import spanner_postgres
from .test_data import (
    service_embedding_1,
    service_embedding_2,
    service_embedding_3,
    kursus_embedding_1,
    kursus_embedding_2,
    kursus_embedding_3,
    faq_embedding_1,
    faq_embedding_2,
    faq_embedding_3,
)
from .utils import get_env_var

pytestmark = pytest.mark.asyncio(scope="module")


@pytest.fixture(scope="module")
def db_project() -> str:
    return get_env_var("DB_PROJECT", "Google Cloud Project")


@pytest.fixture(scope="module")
def db_instance() -> str:
    return get_env_var("DB_INSTANCE", "Spanner Instance")


@pytest.fixture(scope="module")
def db_name() -> str:
    return get_env_var("DB_NAME", "Spanner PG Database")


@pytest.fixture(scope="module")
def create_db(
    db_project: str, db_instance: str, db_name: str
) -> Generator[str, None, None]:
    client = spanner.Client(project=db_project)
    instance = client.instance(db_instance)

    database = instance.database(
        db_name,
        database_dialect=DatabaseDialect.POSTGRESQL,
    )

    database.create()

    yield db_name

    database.drop()
    client.close()


@pytest_asyncio.fixture(scope="module")
async def ds(
    create_db: str,
    db_project: str,
    db_instance: str,
) -> AsyncGenerator[datastore.Client, None]:
    cfg = spanner_postgres.Config(
        kind="spanner-postgres",
        project=db_project,
        instance=db_instance,
        database=create_db,
    )

    ds = await datastore.create(cfg)

    service_ds_path = "../data/service_dummy.csv"
    kursus_ds_path = "../data/kursus_dummy.csv"
    faq_ds_path = "../data/faq_dummy.csv"
    services, kursus_list, faqs = await ds.load_dataset(
        service_ds_path,
        kursus_ds_path,
        faq_ds_path,
    )
    await ds.initialize_data(services, kursus_list, faqs)

    if ds is None:
        raise TypeError("datastore creation failure")

    yield ds

    await ds.close()


def check_file_diff(file_diff):
    assert file_diff["added"] == []
    assert file_diff["removed"] == []
    assert file_diff["changed"] == []
    assert file_diff["columns_added"] == []
    assert file_diff["columns_removed"] == []


async def test_export_dataset(ds: spanner_postgres.Client):
    services, kursus_list, faqs = await ds.export_data()

    service_ds_path = "../data/service_dummy.csv"
    kursus_ds_path = "../data/kursus_dummy.csv"
    faq_ds_path = "../data/faq_dummy.csv"

    service_new_path = "../data/service_dummy.csv.new"
    kursus_new_path = "../data/kursus_dummy.csv.new"
    faq_new_path = "../data/faq_dummy.csv.new"

    await ds.export_dataset(
        services,
        kursus_list,
        faqs,
        service_new_path,
        kursus_new_path,
        faq_new_path,
    )

    diff_services = compare(
        load_csv(open(service_ds_path), "id"), load_csv(open(service_new_path), "id")
    )
    check_file_diff(diff_services)

    diff_kursus = compare(
        load_csv(open(kursus_ds_path), "id"),
        load_csv(open(kursus_new_path), "id"),
    )
    check_file_diff(diff_kursus)

    diff_faq = compare(
        load_csv(open(faq_ds_path), "id"), load_csv(open(faq_new_path), "id")
    )
    check_file_diff(diff_faq)


async def test_search_services(ds: spanner_postgres.Client):
    query_embedding = service_embedding_1
    similarity_threshold = 0.5
    top_k = 3
    res, sql = await ds.search_services(query_embedding, similarity_threshold, top_k)
    assert isinstance(res, list)
    assert sql is not None


async def test_search_kursus(ds: spanner_postgres.Client):
    query_embedding = kursus_embedding_1
    similarity_threshold = 0.5
    top_k = 3
    res, sql = await ds.search_kursus(query_embedding, similarity_threshold, top_k)
    assert isinstance(res, list)
    assert sql is not None


async def test_search_faqs(ds: spanner_postgres.Client):
    query_embedding = faq_embedding_1
    similarity_threshold = 0.5
    top_k = 3
    res, sql = await ds.search_faqs(query_embedding, similarity_threshold, top_k)
    assert isinstance(res, list)
    assert sql is not None

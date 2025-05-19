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
from typing import Any, AsyncGenerator, List

import pytest
import pytest_asyncio
from csv_diff import compare, load_csv  # type: ignore

import models

from .. import datastore
from . import postgres
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
def db_user() -> str:
    return get_env_var("DB_USER", "name of a postgres user")


@pytest.fixture(scope="module")
def db_pass() -> str:
    return get_env_var("DB_PASS", "password for the postgres user")


@pytest.fixture(scope="module")
def db_name() -> str:
    return get_env_var("DB_NAME", "name of a postgres database")


@pytest.fixture(scope="module")
def db_host() -> str:
    return get_env_var("DB_HOST", "ip address of a postgres database")


@pytest_asyncio.fixture(scope="module")
async def ds(
    db_user: str, db_pass: str, db_name: str, db_host: str
) -> AsyncGenerator[datastore.Client, None]:
    cfg = postgres.Config(
        kind="postgres",
        user=db_user,
        password=db_pass,
        database=db_name,
        host=IPv4Address(db_host),
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


async def test_export_dataset(ds: postgres.Client):
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


async def test_search_services(ds: postgres.Client):
    query_embedding = service_embedding_1
    similarity_threshold = 0.5
    top_k = 3
    res, sql = await ds.search_services(query_embedding, similarity_threshold, top_k)
    assert isinstance(res, list)
    assert sql is not None


async def test_search_kursus(ds: postgres.Client):
    query_embedding = kursus_embedding_1
    similarity_threshold = 0.5
    top_k = 3
    res, sql = await ds.search_kursus(query_embedding, similarity_threshold, top_k)
    assert isinstance(res, list)
    assert sql is not None


async def test_search_faqs(ds: postgres.Client):
    query_embedding = faq_embedding_1
    similarity_threshold = 0.5
    top_k = 3
    res, sql = await ds.search_faqs(query_embedding, similarity_threshold, top_k)
    assert isinstance(res, list)
    assert sql is not None

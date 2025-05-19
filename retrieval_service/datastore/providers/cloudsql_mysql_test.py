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

import asyncio
from typing import Any, AsyncGenerator, List

import pymysql
import pytest
import pytest_asyncio
from csv_diff import compare, load_csv  # type: ignore
from google.cloud.sql.connector import Connector

import models

from .. import datastore
from . import cloudsql_mysql
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
    return get_env_var("DB_USER", "name of a mysql user")


@pytest.fixture(scope="module")
def db_pass() -> str:
    return get_env_var("DB_PASS", "password for the mysql user")


@pytest.fixture(scope="module")
def db_project() -> str:
    return get_env_var("DB_PROJECT", "project id for google cloud")


@pytest.fixture(scope="module")
def db_region() -> str:
    return get_env_var("DB_REGION", "region for cloud sql instance")


@pytest.fixture(scope="module")
def db_instance() -> str:
    return get_env_var("DB_INSTANCE", "instance for cloud sql")


@pytest_asyncio.fixture(scope="module")
async def create_db(
    db_user: str, db_pass: str, db_project: str, db_region: str, db_instance: str
) -> AsyncGenerator[str, None]:
    db_name = get_env_var("DB_NAME", "name of a cloud sql mysql database")
    loop = asyncio.get_running_loop()
    connector = Connector(loop=loop)
    project_instance = f"{db_project}:{db_region}:{db_instance}"
    # Database does not exist, create it.
    sys_conn: pymysql.Connection = await connector.connect_async(
        # cloud sql instance connection name
        project_instance,
        "pymysql",
        user=f"{db_user}",
        password=f"{db_pass}",
        db="mysql",
    )
    cursor = sys_conn.cursor()

    cursor.execute(f"drop database if exists {db_name};")
    cursor.execute(f"CREATE DATABASE {db_name};")
    conn: pymysql.Connection = await connector.connect_async(
        # Cloud SQL instance connection name
        project_instance,
        "pymysql",
        user=f"{db_user}",
        password=f"{db_pass}",
        db=f"{db_name}",
    )
    conn.close()
    yield db_name
    cursor.execute(f"drop database if exists {db_name};")
    cursor.close()


@pytest_asyncio.fixture(scope="module")
async def ds(
    create_db: str,
    db_user: str,
    db_pass: str,
    db_project: str,
    db_region: str,
    db_instance: str,
) -> AsyncGenerator[datastore.Client, None]:
    cfg = cloudsql_mysql.Config(
        kind="cloudsql-mysql",
        user=db_user,
        password=db_pass,
        database=create_db,
        project=db_project,
        region=db_region,
        instance=db_instance,
    )
    ds = await datastore.create(cfg)

    airports_ds_path = "../data/airport_dataset.csv"
    amenities_ds_path = "../data/amenity_dataset.csv"
    flights_ds_path = "../data/flights_dataset.csv"
    policies_ds_path = "../data/cymbalair_policy.csv"
    airports, amenities, flights, policies = await ds.load_dataset(
        airports_ds_path,
        amenities_ds_path,
        flights_ds_path,
        policies_ds_path,
    )
    await ds.initialize_data(airports, amenities, flights, policies)

    if ds is None:
        raise TypeError("datastore creation failure")
    yield ds

    await ds.close()


def only_embedding_changed(file_diff):
    return all(
        key == "embedding"
        for change in file_diff["changed"]
        for key in change["changes"]
    )


def check_file_diff(file_diff):
    assert file_diff["added"] == []
    assert file_diff["removed"] == []
    assert file_diff["columns_added"] == []
    assert file_diff["columns_removed"] == []
    assert file_diff["changed"] == [] or only_embedding_changed(file_diff)


async def test_export_dataset(ds: cloudsql_mysql.Client):
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


# =========================
# TEST UNTUK SERVICE, KURSUS, FAQ
# =========================

@pytest.mark.asyncio
async def test_service_kursus_faq(ds: cloudsql_mysql.Client):
    # Dummy data sesuai model Service, Kursus, Faq
    services = [
        models.Service(
            id=1,
            category="Terjemahan",
            title="Terjemah Ijazah",
            description="Penerjemahan ijazah dari/ke Bahasa Indonesia, Arab, atau Inggris.",
            price=75000,
            embedding=[0.1] * 768,
        ),
        models.Service(
            id=2,
            category="Terjemahan",
            title="Terjemah Transkrip Nilai",
            description="Penerjemahan transkrip nilai akademik dari/ke Bahasa Indonesia, Arab, atau Inggris.",
            price=100000,
            embedding=[0.2] * 768,
        ),
    ]
    kursus_list = [
        models.Kursus(
            id=1,
            course_name="Kursus Bahasa Inggris Dasar",
            level="Dasar",
            description="Pengenalan grammar dasar, vocabulary umum, dan percakapan sehari-hari.",
            price=500000,
            start_date="6/1/2025",
            end_date="7/15/2025",
            embedding=[0.3] * 768,
        )
    ]
    faqs = [
        models.Faq(
            id=1,
            category="Pendaftaran",
            title="Bagaimana cara mendaftar kursus?",
            description="Kunjungi https://pusatbahasa.uinjkt.ac.id/pendaftaran, isi formulir online, dan lakukan pembayaran.",
            embedding=[0.4] * 768,
        )
    ]

    # Insert data ke datastore
    await ds.initialize_data(services, kursus_list, faqs)
    exported_services, exported_kursus, exported_faqs = await ds.export_data()

    # Cek hasil export sama dengan data yang diinsert
    assert len(exported_services) == len(services)
    assert len(exported_kursus) == len(kursus_list)
    assert len(exported_faqs) == len(faqs)
    assert exported_services[0].title == services[0].title
    assert exported_kursus[0].course_name == kursus_list[0].course_name
    assert exported_faqs[0].title == faqs[0].title

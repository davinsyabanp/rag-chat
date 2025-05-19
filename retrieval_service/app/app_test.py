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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from google.oauth2 import id_token

import datastore
import models

from . import init_app


@pytest.fixture(scope="module")
def app():
    mock_cfg = MagicMock()
    mock_cfg.clientId = "fake client id"
    app = init_app(mock_cfg)
    if app is None:
        raise TypeError("app did not initialize")
    return app


# =========================
# FAQ TESTS
# =========================

faq_params = [
    pytest.param(
        "get_faq_by_id",
        {"id": 1},
        models.FAQ(
            id=1,
            category="Pendaftaran",
            title="Bagaimana cara mendaftar kursus?",
            description="Silakan kunjungi laman pendaftaran di website kami.",
        ),
        {
            "id": 1,
            "category": "Pendaftaran",
            "title": "Bagaimana cara mendaftar kursus?",
            "description": "Silakan kunjungi laman pendaftaran di website kami.",
        },
        id="faq_by_id",
    ),
]


@pytest.mark.parametrize("method_name, params, mock_return, expected", faq_params)
@patch.object(datastore, "create")
def test_get_faq(m_datastore, app, method_name, params, mock_return, expected):
    with TestClient(app) as client:
        with patch.object(
            m_datastore.return_value,
            method_name,
            AsyncMock(return_value=(mock_return, None)),
        ) as mock_method:
            response = client.get("/faq", params=params)
    assert response.status_code == 200
    res = response.json()
    output = res["results"]
    assert output == expected
    assert models.FAQ.model_validate(output)


# =========================
# KURSUS TESTS
# =========================

kursus_params = [
    pytest.param(
        "get_kursus_by_id",
        {"id": 1},
        models.Kursus(
            id=1,
            course_name="Bahasa Inggris Dasar",
            level="Beginner",
            description="Kursus untuk pemula.",
            price=500000,
            start_date="2024-07-01",
            end_date="2024-08-01",
        ),
        {
            "id": 1,
            "course_name": "Bahasa Inggris Dasar",
            "level": "Beginner",
            "description": "Kursus untuk pemula.",
            "price": 500000,
            "start_date": "2024-07-01",
            "end_date": "2024-08-01",
        },
        id="kursus_by_id",
    ),
]


@pytest.mark.parametrize("method_name, params, mock_return, expected", kursus_params)
@patch.object(datastore, "create")
def test_get_kursus(m_datastore, app, method_name, params, mock_return, expected):
    with TestClient(app) as client:
        with patch.object(
            m_datastore.return_value,
            method_name,
            AsyncMock(return_value=(mock_return, None)),
        ) as mock_method:
            response = client.get("/kursus", params=params)
    assert response.status_code == 200
    res = response.json()
    output = res["results"]
    assert output == expected
    assert models.Kursus.model_validate(output)


# =========================
# SERVICE TESTS
# =========================

service_params = [
    pytest.param(
        "get_service_by_id",
        {"id": 1},
        models.Service(
            id=1,
            category="Penerjemahan",
            title="Jasa Penerjemahan Dokumen",
            description="Layanan penerjemahan dokumen resmi.",
            price=250000,
        ),
        {
            "id": 1,
            "category": "Penerjemahan",
            "title": "Jasa Penerjemahan Dokumen",
            "description": "Layanan penerjemahan dokumen resmi.",
            "price": 250000,
        },
        id="service_by_id",
    ),
]


@pytest.mark.parametrize("method_name, params, mock_return, expected", service_params)
@patch.object(datastore, "create")
def test_get_service(m_datastore, app, method_name, params, mock_return, expected):
    with TestClient(app) as client:
        with patch.object(
            m_datastore.return_value,
            method_name,
            AsyncMock(return_value=(mock_return, None)),
        ) as mock_method:
            response = client.get("/service", params=params)
    assert response.status_code == 200
    res = response.json()
    output = res["results"]
    assert output == expected
    assert models.Service.model_validate(output)

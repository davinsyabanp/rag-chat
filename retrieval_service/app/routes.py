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

from typing import Any, Mapping, Optional

from fastapi import APIRouter, HTTPException, Request
from google.auth.transport import requests  # type:ignore
from google.oauth2 import id_token  # type:ignore
from langchain_core.embeddings import Embeddings

import datastore

routes = APIRouter()


def _ParseUserIdToken(headers: Mapping[str, Any]) -> Optional[str]:
    """Parses the bearer token out of the request headers."""
    # authorization_header = headers.lower()
    user_id_token_header = headers.get("User-Id-Token")
    if not user_id_token_header:
        raise Exception("no user authorization header")

    parts = str(user_id_token_header).split(" ")
    if len(parts) != 2 or parts[0] != "Bearer":
        raise Exception("Invalid ID token")

    return parts[1]


async def get_user_info(request):
    headers = request.headers
    token = _ParseUserIdToken(headers)
    try:
        id_info = id_token.verify_oauth2_token(
            token, requests.Request(), audience=request.app.state.client_id
        )

        return {
            "user_id": id_info.get("sub"),
            "user_name": id_info.get("name"),
            "user_email": id_info.get("email"),
        }

    except Exception as e:  # pylint: disable=broad-except
        print(e)


@routes.get("/")
async def root():
    return {"message": "Hello World"}

# Endpoint untuk mengambil data layanan berdasarkan id
@routes.get("/services")
async def get_service(
    request: Request,
    id: Optional[int] = None,
    category: Optional[str] = None,
):
    ds: datastore.Client = request.app.state.datastore
    if id:
        results, sql = await ds.get_service_by_id(id)
    elif category:
        results, sql = await ds.get_service_by_category(category)
    else:
        raise HTTPException(
            status_code=422,
            detail="Request requires query params: service id or category",
        )
    return {"results": results, "sql": sql}


# Endpoint pencarian layanan berbasis embedding
@routes.get("/services/search")
async def search_services(
    request: Request,
    query: str,
    top_k: int = 5,
):
    ds: datastore.Client = request.app.state.datastore
    embed_service: Embeddings = request.app.state.embed_service
    query_embedding = embed_service.embed_query(query)
    results, sql = await ds.services_search(query_embedding, 0.5, top_k)
    return {"results": results, "sql": sql}


# Endpoint untuk mengambil data kursus berdasarkan id
@routes.get("/courses")
async def get_course(
    request: Request,
    id: Optional[int] = None,
    level: Optional[str] = None,
):
    ds: datastore.Client = request.app.state.datastore
    if id:
        results, sql = await ds.get_course_by_id(id)
    elif level:
        results, sql = await ds.get_course_by_level(level)
    else:
        raise HTTPException(
            status_code=422,
            detail="Request requires query params: course id or level",
        )
    return {"results": results, "sql": sql}


# Endpoint pencarian kursus berbasis embedding
@routes.get("/courses/search")
async def search_courses(
    request: Request,
    query: str,
    top_k: int = 5,
):
    ds: datastore.Client = request.app.state.datastore
    embed_service: Embeddings = request.app.state.embed_service
    query_embedding = embed_service.embed_query(query)
    results, sql = await ds.courses_search(query_embedding, 0.5, top_k)
    return {"results": results, "sql": sql}


# Endpoint untuk mengambil FAQ berdasarkan id/kategori
@routes.get("/faqs")
async def get_faq(
    request: Request,
    id: Optional[int] = None,
    category: Optional[str] = None,
):
    ds: datastore.Client = request.app.state.datastore
    if id:
        results, sql = await ds.get_faq_by_id(id)
    elif category:
        results, sql = await ds.get_faq_by_category(category)
    else:
        raise HTTPException(
            status_code=422,
            detail="Request requires query params: faq id or category",
        )
    return {"results": results, "sql": sql}


# Endpoint pencarian FAQ berbasis embedding
@routes.get("/faqs/search")
async def search_faqs(
    request: Request,
    query: str,
    top_k: int = 5,
):
    ds: datastore.Client = request.app.state.datastore
    embed_service: Embeddings = request.app.state.embed_service
    query_embedding = embed_service.embed_query(query)
    results, sql = await ds.faqs_search(query_embedding, 0.5, top_k)
    return {"results": results, "sql": sql}

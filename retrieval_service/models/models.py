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

import ast
import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


class Service(BaseModel):
    id: int
    category: str
    title: str
    description: str
    price: int


class Kursus(BaseModel):
    id: int
    course_name: str
    level: str
    description: str
    price: int
    start_date: str  # format: MM/DD/YYYY
    end_date: str    # format: MM/DD/YYYY


class FAQ(BaseModel):
    id: int
    category: str
    title: str
    description: str

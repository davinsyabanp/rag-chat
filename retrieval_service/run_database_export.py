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

import datastore
from app import parse_config


async def main():
    cfg = parse_config("config.yml")
    ds = await datastore.create(cfg.datastore)

    services, kursus_list, faqs = await ds.export_data()

    await ds.close()

    services_new_path = "../data/service_dummy.csv.new"
    kursus_new_path = "../data/kursus_dummy.csv.new"
    faqs_new_path = "../data/faq_dummy.csv.new"

    await ds.export_dataset(
        services,
        kursus_list,
        faqs,
        services_new_path,
        kursus_new_path,
        faqs_new_path,
    )

    print("database export done.")


if __name__ == "__main__":
    asyncio.run(main())

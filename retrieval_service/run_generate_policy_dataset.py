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

import time

import pandas as pd
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app import EMBEDDING_MODEL_NAME


def main() -> None:
    faqs_ds_path = "../data/faq_dummy.csv.new"

    chunked = text_split(_FAQ)
    data_embeddings = vectorize(chunked)
    data_embeddings.to_csv(faqs_ds_path, index=True, index_label="id")

    print("Done generating FAQ dataset.")


def text_split(data):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(data)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=30,
        length_function=len,
    )
    splits = text_splitter.split_documents(md_header_splits)

    chunked = [{"content": s.page_content} for s in splits]
    return chunked


def vectorize(chunked):
    embed_service = VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    def retry_with_backoff(func, *args, retry_delay=5, backoff_factor=2, **kwargs):
        max_attempts = 3
        retries = 0
        for i in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"error: {e}")
                retries += 1
                wait = retry_delay * (backoff_factor**retries)
                print(f"Retry after waiting for {wait} seconds...")
                time.sleep(wait)

    batch_size = 5
    for i in range(0, len(chunked), batch_size):
        request = [x["content"] for x in chunked[i : i + batch_size]]
        response = retry_with_backoff(embed_service.embed_documents, request)
        # Store the retrieved vector embeddings for each chunk back.
        for x, e in zip(chunked[i : i + batch_size], response):
            x["embedding"] = e

    data_embeddings = pd.DataFrame(chunked)
    data_embeddings.head()
    return data_embeddings


_FAQ = """# FAQ Pusat Pengembangan Bahasa
## Pendaftaran
Bagaimana cara mendaftar kursus? Kunjungi https://pusatbahasa.uinjkt.ac.id/pendaftaran, isi formulir online, dan lakukan pembayaran.
Apakah ada batas waktu pendaftaran? Pendaftaran ditutup 2 minggu sebelum kursus dimulai.
## Pembayaran
Metode pembayaran apa yang diterima? Transfer bank (BCA/Mandiri), e-wallet (GoPay, OVO), dan kartu kredit.
Apakah bisa dicicil? Maksimal 2 kali cicilan tanpa bunga dengan minimum pembayaran 50% di awal.
## Sertifikat
Apakah ada sertifikat setelah selesai kursus? Ya, sertifikat resmi Pusat Pengembangan Bahasa UIN Syarif Hidayatullah Jakarta.
## Refund
Bagaimana kebijakan refund? Refund 75% jika pembatalan > 7 hari sebelum mulai, 50% jika 3–7 hari, dan 0% jika < 3 hari.
## Operasional
Jam operasional Pusat Pengembangan Bahasa? Senin–Jumat: 08:00–16:00, Sabtu: 08:00–12:00.
## Kontak
Bagaimana menghubungi admin? Email ke ppb@uinjkt.ac.id atau telepon ke (021) 7828 1234.
## Teknis
Apakah kursus online juga tersedia? Ya, tersedia via Zoom dengan link khusus peserta.
Bagaimana cara akses materi online? Login ke portal e-learning dengan akun yang diberikan setelah pembayaran.
## Program
Apakah ada program beasiswa? Saat ini belum, tetapi cek pengumuman rutin di website.
## Level
Bagaimana menentukan level kursus? Ikuti placement test online gratis sebelum daftar.
## Modul
Apa saja materi yang dipelajari? Grammar, vocabulary, listening, speaking, reading, writing sesuai silabus.
## Perubahan
Bisakah mengubah jadwal setelah daftar? Ya, dengan surcharge 100K jika permintaan 3 hari sebelum kelas.
## Diskon
Apakah ada diskon group? Diskon 10% untuk kelompok minimal 3 orang.
"""

if __name__ == "__main__":
    main()

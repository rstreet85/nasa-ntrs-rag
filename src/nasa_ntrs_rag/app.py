import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TypedDict

import bm25s
import httpx
import numpy as np
import pymupdf4llm

LOG_FILENAME = 'test.log'
DOWNLOAD_DB_PATH = Path('articles.db')
DOWNLOAD_PATH = Path('pdf')
NTRS_URL = 'https://ntrs.nasa.gov'
NTRS_PUBSEARCH_URL = 'https://ntrs.nasa.gov/api/pubspace/search'
DOWNLOAD_DB_SCHEMA = '''
    CREATE TABLE IF NOT EXISTS articles_status (
        id INTEGER PRIMARY KEY,
        status TEXT NOT NULL,
        url TEXT NOT NULL,
        filename TEXT NOT NULL,
        title TEXT NOT NULL
        );'''
METADATA_REQ_PARAMS = {
    'subjectCategory': [
        'Space Communications, Spacecraft Communications, Command and Tracking'
    ]
}
TEST_QUERY = 'What factors affect satellite altimetry measurement quality'
K = 3

logger = logging.getLogger(__name__)

class Chunk(TypedDict):
    '''Class to use a TypedDict to store chunk data.'''
    chunk_id: str
    document_id: int
    page_number: int
    title: str
    url: str
    text: str


def init_download_db() -> None:
    '''Initialize the article database.'''
    with closing(sqlite3.connect(DOWNLOAD_DB_PATH)) as conn:
        with conn:
            cursor = conn.cursor()
            cursor.execute(DOWNLOAD_DB_SCHEMA)
            print('Downloads database initialized')


def get_metadata(cursor: sqlite3.Cursor) -> None:
    '''Search all articles that match requested category code and have PDF downloads. Save IDs in db.'''
    try:
        response = httpx.post(NTRS_PUBSEARCH_URL, json=METADATA_REQ_PARAMS)
        response.raise_for_status()

        pub_list = response.json()['results']
        print('Metadata response received')
        
        for pub in pub_list:
            if pub['downloadsAvailable'] and pub['downloads']: # There are edge cases where downloads = true and list of downloads is empty
                article_id = int(pub['id'])
                url = pub['downloads'][0]['links']['pdf']
                title = pub['title']
                
                cursor.execute('''
                    INSERT OR IGNORE INTO articles_status (
                    id,
                    status,
                    url,
                    filename,
                    title
                    )
                    VALUES (?, ?, ?, ?, ?);''',
                    (article_id, 'pending', url, url.split('/')[-1], title))

        print('Metadata response processed')
    except httpx.HTTPStatusError as exc:
        print(f"Error status {exc.response.status_code} received when requesting metadata")
    except httpx.RequestError as exc:
        print(f"Request error for metadata: {exc}")
    except httpx.HTTPError as exc:
        print(f"Unexpected HTTP error: {exc}")


def download_files(cursor: sqlite3.Cursor) -> None:
    '''Read list of all files undownloaded files and download them.'''
    print('Beginning file downloads')
    cursor.execute('SELECT id, status, url, filename, title FROM articles_status WHERE status != ?', ('downloaded',))
    docs = cursor.fetchall()

    for doc in docs:
        download_pdf(cursor, doc[0], doc[2], doc[3])

    print('Downloading files complete')


def download_pdf(cursor: sqlite3.Cursor, article_id: int, url: str, filename: str) -> None:
    '''Take article ID, url, and filename and download the PDF, update the status in the database.'''
    print(f"Downloading file {filename}")

    try:
        response = httpx.get(f"{NTRS_URL}{url}")
        response.raise_for_status()

        download_filepath = DOWNLOAD_PATH / filename
        with download_filepath.open('wb') as file:
            file.write(response.content)
            cursor.execute('UPDATE articles_status SET status = ? WHERE id = ?', ('downloaded', article_id))
            print(f"{filename} downloaded")

    except httpx.HTTPError as exc:
        cursor.execute('UPDATE articles_status SET status = ? WHERE id = ?', ('failed', article_id))
        print(f"{filename} download failed - {exc}")


def extract_pdfs(cursor: sqlite3.Cursor) -> list[Chunk]:
    '''Create list of downloaded PDFs from the database, extract text from each.'''
    print('Extracting text from PDFs')
    cursor.execute('SELECT id, status, url, filename, title FROM articles_status WHERE status = ?', ('downloaded',))
    saved_docs = cursor.fetchall()
    combined_chunks: list[Chunk] = []

    for document in saved_docs:
        filepath = DOWNLOAD_PATH / document[3]
        doc_chunks = pymupdf4llm.to_markdown(filepath, page_chunks=True)

        for chunk_number, chunk in enumerate(doc_chunks):
            combined_chunks.append({
                'chunk_id': f"{document[0]}-{chunk_number}",
                'document_id': document[0],
                'page_number': chunk['metadata']['page_number'],
                'title': document[4],
                'url': f"{NTRS_URL}{document[2]}",
                'text': chunk['text']
            })

    return combined_chunks


def doc_search_lexical(chunks: list[Chunk], query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    '''Accept text corpus, search query, and number of top results, returns top k results and scores.'''
    print('Searching combined text')
    corpus = [chunk['text'] for chunk in chunks]
    tokenized_corp = bm25s.tokenize(corpus, stopwords='english')
    tokenized_query = bm25s.tokenize(query)

    retriever = bm25s.BM25()
    retriever.index(tokenized_corp)
    results, scores = retriever.retrieve(tokenized_query, corpus=chunks, k=k)
    return results, scores


def main() -> None:
    logging.basicConfig(filename=LOG_FILENAME, encoding='utf-8', level=logging.DEBUG)

    init_download_db()

    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(DOWNLOAD_DB_PATH)) as conn:
        with conn:
            cursor = conn.cursor()
            get_metadata(cursor)
            download_files(cursor)

            combined_chunks = extract_pdfs(cursor)

            results, scores = doc_search_lexical(combined_chunks, TEST_QUERY, K)
            for i in range(results.shape[1]):
                result_chunk = results[0, i]
                print(f"Document Title: {result_chunk['title']}, Page {result_chunk['page_number']}")
                print(f"Text:\n{result_chunk['text']}\n\n")
                '''
                print(f"Chunk Score: {scores[0, i]}")
                print(f"Chunk ID: {result_chunk['chunk_id']}")
                print(f"Document ID: {result_chunk['document_id']}, Page: {result_chunk['page_number']}")
                print(f"Document Title: {result_chunk['title']}")
                print(f"Document URL: {result_chunk['url']}")
                print(f"Chunk Text:\n{result_chunk['text']}\n\n")
                '''

            print('All PDFs examined')


if __name__ == '__main__':
    main()
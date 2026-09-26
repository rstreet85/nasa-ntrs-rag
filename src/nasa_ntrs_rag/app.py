import bm25s
import httpx
from pathlib import Path
import pymupdf4llm
import sqlite3

DOWNLOAD_DB_URL = 'articles.db'
DOWNLOAD_URL = 'pdf/'
NTRS_URL = 'https://ntrs.nasa.gov'
NTRS_PUBSEARCH_URL = 'https://ntrs.nasa.gov/api/pubspace/search'

DOWNLOAD_DB_SCHEMA = 'CREATE TABLE IF NOT EXISTS articles_status (id INTEGER PRIMARY KEY, status TEXT NOT NULL, url TEXT NOT NULL, filename TEXT NOT NULL, title TEXT NOT NULL);'

METADATA_REQ_PARAMS = {
    'subjectCategory' : [
        'Space Communications, Spacecraft Communications, Command and Tracking'
    ]
}

TEST_QRY = 'size of keep-out-sphere'
K = 3

# Init the db for tracking article IDs and download status
def init_download_db() -> None:
    with sqlite3.connect(DOWNLOAD_DB_URL) as conn:
        cursor = conn.cursor()
        cursor.execute(DOWNLOAD_DB_SCHEMA)
        print('Downloads database initialized')

# Search all articles that both match requested category code and have PDF downloads. Save IDs in db
def get_metadata(cursor: sqlite3.Cursor) -> None:
    try:
        response = httpx.post(NTRS_PUBSEARCH_URL, json=METADATA_REQ_PARAMS)
        response.raise_for_status()

        pub_list = response.json()['results']
        print('Metadata response received')
        
        for pub in pub_list:
            if pub['downloadsAvailable'] and pub['downloads']: # There are edge cases where downloads = true and list of downloads is empty
                id = int(pub['id'])
                url = pub['downloads'][0]['links']['pdf']
                title = pub['title']
                
                cursor.execute('INSERT OR IGNORE INTO articles_status (id, status, url, filename, title) VALUES (?, ?, ?, ?, ?);', (id, 'pending', url, url.split('/')[-1], title))

        print('Metadata response processed')
    except httpx.HTTPError as ex:
        print(f'Error code {ex.response.status_code} while requesting metadata')

# Read list of all files that haven't been downloaded and download them
def download_files(cursor: sqlite3.cursor) -> None:
    print(f'Beginning file downloads')
    cursor.execute('SELECT id, status, url, filename, title FROM articles_status WHERE status != ?', ('downloaded',))
    docs = cursor.fetchall()

    for doc in docs:
        download_pdf(cursor, doc[0], doc[2], doc[3])

    print(f'Downloading files complete')

# Download an individual PDF
def download_pdf(cursor: sqlite3.cursor, id: int, url: str, filename: str) -> None:
    print(f'Downloading file {filename}')

    try:
        response = httpx.get(f'{NTRS_URL}{url}')
        response.raise_for_status()

        with open(f'{DOWNLOAD_URL}{filename}', 'wb') as file:
            file.write(response.content)
            cursor.execute('UPDATE articles_status SET status = ? WHERE id = ?', ('downloaded', id))
            print(f'{filename} downloaded')

    except httpx.HTTPError as ex:
        cursor.execute('UPDATE articles_status SET status = ? WHERE id = ?', ('failed', id))
        print(f'{filename} download failed with error code {ex.response.status_code}')

# Loop through downloaded files and extract text using PyMuPDF4lLM
def extract_pdfs(cursor: sqlite3.cursor) -> list[dict]:
    print('Extracting text from PDFs')
    cursor.execute('SELECT id, status, url, filename, title FROM articles_status WHERE status = ?', ('downloaded',))
    saved_docs = cursor.fetchall()
    combined_chunks = []

    for document in saved_docs:
        filepath = Path(DOWNLOAD_URL) / document[3]
        doc_chunks = pymupdf4llm.to_markdown(filepath, page_chunks=True)

        for chunk_number, chunk in enumerate(doc_chunks):
            combined_chunks.append({
                'chunk_id': f'{document[0]}-{chunk_number}',
                'document_id': document[0],
                'page_number': chunk['metadata']['page_number'],
                'title': document[4],
                'url': f'{NTRS_URL}{document[2]}',
                'text': chunk['text']
            })

    return combined_chunks

# Search all text for a query
def search_docs(chunks: list[dict], qry: str, k: int) -> None:
    print('Searching combined text')
    corpus = [chunk['text'] for chunk in chunks]
    tokenized_corp = bm25s.tokenize(corpus, stopwords='english')
    tokenized_qry = bm25s.tokenize(qry)

    retriever = bm25s.BM25()
    retriever.index(tokenized_corp)
    results, scores = retriever.retrieve(tokenized_qry, corpus=chunks, k=k)

    for i in range(results.shape[1]):
        result_chunk = results[0, i]
        print(f'Chunk Score: {scores[0, i]}')
        print(f'Chunk ID: {result_chunk['chunk_id']}')
        print(f'Document ID: {result_chunk['document_id']}, Page: {result_chunk['page_number']}')
        print(f'Document Title: {result_chunk['title']}')
        print(f'Document URL: {result_chunk['url']}')
        print(f'Chunk Text:\n{result_chunk['text']}\n\n')

# Document ingestion
init_download_db()

with sqlite3.connect(DOWNLOAD_DB_URL) as conn:
    cursor = conn.cursor()
    get_metadata(cursor)
    download_files(cursor)

    combined_chunks = extract_pdfs(cursor)

    search_docs(combined_chunks, TEST_QRY, K)

    print('All PDFs examined')
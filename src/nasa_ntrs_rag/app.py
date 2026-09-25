import bm25s
import httpx
#import json
from pathlib import Path
import pymupdf4llm
import sqlite3

DOWNLOAD_DB_URL = 'articles.db'
DOWNLOAD_URL = 'pdf/'
NTRS_URL = 'https://ntrs.nasa.gov'
NTRS_PUBSEARCH_URL = 'https://ntrs.nasa.gov/api/pubspace/search'

DOWNLOAD_DB_SCHEMA = 'CREATE TABLE IF NOT EXISTS articles_status (id INTEGER PRIMARY KEY, status TEXT NOT NULL, url TEXT NOT NULL);'

METADATA_REQ_PARAMS = {
    'subjectCategory' : [
        'Space Communications, Spacecraft Communications, Command and Tracking'
    ]
}

TEST_QRY = 'what is dpod'

# Init the db for tracking article IDs and download status
def init_download_db() -> None:
    with sqlite3.connect(DOWNLOAD_DB_URL) as conn:
        cursor = conn.cursor()
        cursor.execute(DOWNLOAD_DB_SCHEMA)
        print('Downloads database created')

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
                
                cursor.execute('INSERT INTO articles_status (id, status, url) VALUES (?, ?, ?);', (id, 'pending', url))

        print('Metadata response processed')
    except httpx.HTTPError as ex:
        print(f'Error code {ex.response.status_code} while requesting metadata')

# Read list of all files that haven't been downloaded yet
def download_files(cursor: sqlite3.cursor) -> None:
    cursor.execute('SELECT id, status, url FROM articles_status WHERE status != ?', ('downloaded',))
    docs = cursor.fetchall()
    print(f'Beginning file downloads')

    for doc in docs:
        download_pdf(cursor, doc[0], doc[2])

    print(f'Downloading files complete')

# Download an individual PDF
def download_pdf(cursor: sqlite3.cursor, id: int, url: str) -> None:
    filename = url.split('/')[-1]
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

# Search a document's text for a query
def search_doc(corpus: list, qry: str, k: int) -> None:
    print('Searching document')
    tokenized_corp = bm25s.tokenize(corpus, stopwords='english')
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(tokenized_corp)

    tokenized_qry = bm25s.tokenize(TEST_QRY)

    results, scores = retriever.retrieve(tokenized_qry, k=k)
    print(f'The top result is \n{results[0]}')

# Document ingestion
init_download_db()

with sqlite3.connect(DOWNLOAD_DB_URL) as conn:
    cursor = conn.cursor()
    get_metadata(cursor)
    download_files(cursor)

# Text extraction
downloaded_pdfs = list(Path(f'{DOWNLOAD_URL}').glob('*.pdf'))
print('Extracting text from PDFs')

for pdf in downloaded_pdfs:
    #markdown = pymupdf4llm.to_markdown(pdf)
    #print(f'Markdown: \n {markdown}')
    print('Chunking and searching document')
    chunks = pymupdf4llm.to_markdown(pdf, page_chunks=True)

    search_doc([chunk['text'] for chunk in chunks], TEST_QRY, 3)
    '''for chunk in chunks:
        print('=' * 80)
        print(f'Page: {chunk['metadata']['page_number']}')
        print(chunk['text'])'''

print('All PDFs examined')
import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TypedDict

import bm25s
import httpx
import numpy as np
import pymupdf
import pymupdf4llm

LOG_FILENAME = Path('test.log')
DOWNLOAD_DB_PATH = Path('articles.db')
DOWNLOAD_PATH = Path('pdf')
INDEX_PATH = Path('ntrs_index')
NTRS_URL = 'https://ntrs.nasa.gov'
NTRS_PUBSEARCH_URL = 'https://ntrs.nasa.gov/api/pubspace/search'
DOWNLOAD_DB_SCHEMA = '''
    CREATE TABLE IF NOT EXISTS articles_status (
        id INTEGER PRIMARY KEY,
        status TEXT NOT NULL,
        url TEXT NOT NULL,
        filename TEXT NOT NULL,
        title TEXT NOT NULL
        );
    '''
METADATA_REQ_PARAMS = {
    'subjectCategory': [
        'Space Communications, Spacecraft Communications, Command and Tracking'
    ]
}
TEST_QUERIES = [
    'Is the Morse-Lyapunov controller fuel optimal',
    'What are the parameters of TIRS-2',
    'What class receivers are DORIS satellites equipped with',
    'What is a Luby transform',
    'What is the best performing centrality measure of node clusters',
    'Where is the DSS-25 station located',
    'What is the role of polyvinyl alcohol in Bi electroplating'
]
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
            logger.info('Downloads database initialized')


def get_metadata(client: httpx.Client, cursor: sqlite3.Cursor) -> None:
    '''Search all articles that match requested category code and have PDF downloads. Save IDs in db.'''
    insert_count = 0
    skip_count = 0
    try:
        response = client.post(NTRS_PUBSEARCH_URL, json=METADATA_REQ_PARAMS)
        response.raise_for_status()

        # Validate JSON from NTRS metadata service here
        try:
            metadata_payload = response.json()
        except ValueError as exc:
            raise RuntimeError('Metadata request failed - Invalid JSON: ') from exc

        if not isinstance(metadata_payload, dict):
            raise RuntimeError('Metadata request is not JSON')

        if 'results' not in metadata_payload or not isinstance(metadata_payload['results'], list):
            raise RuntimeError('Metadata response is missing valid results list')

        # Parse results if valid JSON is received
        pub_list = metadata_payload['results']
        logger.info('Metadata response received, %i articles found', len(pub_list))

        for pub in pub_list:

            # Validate document metadata
            if not isinstance(pub, dict):
                logger.error('Document metadata is not in valid JSON format')
                continue

            if 'downloadsAvailable' not in pub or 'downloads' not in pub:
                logger.error('Document metadata missing download information')
                continue

            if not isinstance(pub['downloads'], list):
                logger.error('Document download metadata is not in expected format')
                continue

            # Confirm that article has downloads and that download list is not empty
            if pub['downloadsAvailable']:
                download_list, article_id, url, filename, title = None, None, None, None, None

                # Check for empty list after checking if the service indicates downloads are available
                try:
                    download_list = pub['downloads']
                except ValueError:
                    logger.error('Article metadata missing list of downloads')
                    continue

                # Check that article has all relevant fields.
                try:
                    article_id = int(pub['id'])
                except KeyError:
                    logger.error('Article metadata missing field article_id')
                except ValueError:
                    logger.error('Article ID in metadata cannot be converteed to integer. Value=%s', pub['id'])

                try:
                    url = download_list[0]['links']['pdf']
                    filename = url.split('/')[-1]
                except IndexError:
                    logger.error('Article metadata has empty downloads list')
                except KeyError:
                    logger.error('Article metadata missing field url')

                try:
                    title = pub['title']
                except KeyError:
                    logger.error('Article metadata missing field title')

                # Add required document fields to the status table
                if article_id and url and filename and title:
                    try: 
                        cursor.execute('''
                            INSERT INTO articles_status (
                                id,
                                status,
                                url,
                                filename,
                                title
                            )
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(id) DO NOTHING;''',
                            (article_id, 'pending', url, filename, title))

                        if cursor.rowcount == 1:
                            insert_count += 1
                            logger.info('Updated metadata for document ID: %i', article_id)
                        elif cursor.rowcount == 0:
                            skip_count += 1
                            logger.info('Skipped updating metadata for document ID: %i', article_id)

                    except sqlite3.Error:
                        logger.exception('Unexpected failure to update metadata for document ID: %i', article_id)

                else:
                    logger.error('Invalid values for article metadata. Article skipped')

        logger.info('Found %i files available for download. Updated download status for %i files, %i files not updated',
                    insert_count + skip_count, insert_count, skip_count)
    except httpx.HTTPStatusError as exc:
        logger.exception('Error status %s received when requesting metadata', exc.response.status_code)
        raise
    except httpx.RequestError as exc:
        logger.exception('Request error for metadata')
        raise
    except httpx.HTTPError as exc:
        logger.exception('Unexpected HTTP error')
        raise


def download_files(client: httpx.Client, cursor: sqlite3.Cursor) -> None:
    '''Read list of all files undownloaded files and download them.'''
    logger.info('Beginning file downloads')
    cursor.execute('SELECT id, status, url, filename, title FROM articles_status WHERE status != ?',
                ('downloaded',))
    docs = cursor.fetchall()

    download_count = 0
    failed_count = 0
    for doc in docs:
        if download_pdf(client, cursor, doc[0], doc[2], doc[3]):
            download_count += 1
        else:
            failed_count += 1

    logger.info('Downloading files complete')
    logger.info('%i total files. Downloaded %i files, failed to download %i files',
                download_count + failed_count, download_count, failed_count)


def download_pdf(client: httpx.Client, cursor: sqlite3.Cursor, article_id: int, url: str, filename: str) -> bool:
    '''Take article ID, url, and filename and download the PDF, update the status in the database.'''
    logger.info('Downloading file ID: %i', article_id)

    try:
        response = client.get(f"{NTRS_URL}{url}")
        response.raise_for_status()

        download_filepath = DOWNLOAD_PATH / filename
        with download_filepath.open('wb') as file:
            file.write(response.content)
        
        cursor.execute('UPDATE articles_status SET status = ? WHERE id = ?', ('downloaded', article_id))
        
        logger.info('File ID: %i downloaded', article_id)

    except httpx.HTTPError as exc:
        cursor.execute('UPDATE articles_status SET status = ? WHERE id = ?', ('failed', article_id))
        logger.exception('File ID: %i download failed', article_id)

        return False

    except (OSError, PermissionError) as exc:
        cursor.execute('UPDATE articles_status SET status = ? WHERE id = ?', ('failed', article_id))
        logger.exception('File ID: %i download failed due to system error', article_id)

        return False

    return True


def extract_pdfs(cursor: sqlite3.Cursor) -> list[Chunk]:
    '''Create list of downloaded PDFs from the database, extract text from each.'''
    # Add extraction metrics here to track performance
    combined_chunks: list[Chunk] = []
    attempted_doc_count = 0
    extracted_count = 0
    empty_doc_count = 0
    failed_count = 0
    total_chunk_count = 0

    # Pull list of all documents where the PDF has been downloaded
    cursor.execute('SELECT id, status, url, filename, title FROM articles_status WHERE status = ?',
                ('downloaded',))
    saved_docs = cursor.fetchall()
    logger.info('Attempting to extract chunks from %i PDFs', len(saved_docs))

    # Individual document extraction
    for document in saved_docs:
        attempted_doc_count += 1
        doc_chunks = None

        # Read PDF file and pull available chunks
        try:
            filepath = DOWNLOAD_PATH / document[3]
            doc_chunks = pymupdf4llm.to_markdown(filepath, page_chunks=True)

            logger.info('Extracting chunks from file ID: %i', document[0])
        except OSError:
            failed_count += 1
            logger.exception('Failed to access file ID: %i', document[0])
        except pymupdf.FileNotFoundError:
            failed_count += 1
            logger.exception('Failed to find file ID: %i', document[0])
        except pymupdf.EmptyFileError:
            failed_count += 1
            logger.exception('File ID: %i is empty', document[0])
        except pymupdf.FileDataError:
            failed_count += 1
            logger.exception('Unable to read file ID: %i', document[0])
        except TypeError:
            failed_count += 1
            logger.exception('Unable to read file ID: %i, it could be encrypted', document[0])
        except Exception:
            failed_count += 1
            logger.exception('Unexpected exception while reading file ID: %i', document[0])

        # Add chunks and accompanying metadata to list of combined chunks
        if doc_chunks is not None:
            extracted_count += 1
            document_chunks = []

            # May want to set start=1 as the chunk number is used to create ID. Will also need to update pytest files
            for chunk_number, chunk in enumerate(doc_chunks):
                document_chunks.append({
                    'chunk_id': f"{document[0]}-{chunk_number}",
                    'document_id': document[0],
                    'page_number': chunk['metadata']['page_number'],
                    'title': document[4],
                    'url': f"{NTRS_URL}{document[2]}",
                    'text': chunk['text']
                })

                total_chunk_count += len(document_chunks)
                combined_chunks.extend(document_chunks)
                logger.info('Added chunks from file ID: %i', document[0])
        else:
            empty_doc_count +=1
            logger.info('No chunks added from file ID: %i', document[0])

    logger.info('Attempted to extract text from %i documents', attempted_doc_count)
    logger.info('Extracted text from %i documents, %i documents were blank, %i documents failed extraction',
                extracted_count, empty_doc_count, failed_count)
    logger.info('Extracted a total of %i chunks from documents', total_chunk_count)
    return combined_chunks


def build_index(chunks: list[Chunk]) -> bm25s.BM25:
    '''Build the BM25 index from extracted chunks'''
    
    # Remove chunks with empty text
    raw_chunk_count = len(chunks)
    chunks = [chunk for chunk in chunks if chunk['text'].strip()]
    logger.info('Removed %i chunks before building index', raw_chunk_count - len(chunks))

    # Check that there exists at least one chunk before building
    if not chunks:
        raise ValueError('Cannot build index from empty corpus')

    logger.info('Building BM25 index from %i chunks', len(chunks))
    corpus = [chunk['text'] for chunk in chunks]
    tokenized_corpus = bm25s.tokenize(corpus, stopwords='english')
    retriever = bm25s.BM25(corpus=chunks)
    retriever.index(tokenized_corpus)
    retriever.save(INDEX_PATH, corpus=chunks)

    logger.info('BM25 index built')
    return retriever


def doc_search_lexical(retriever: bm25s.BM25, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    '''Accept the BM25 index, chunks, query, and k. Return seach results'''
    logger.info('Searching combined text')
    tokenized_query = bm25s.tokenize(query, stopwords='english')

    results, scores = retriever.retrieve(tokenized_query, k=k)
    return results, scores


def main() -> None:
    logging.basicConfig(filename=LOG_FILENAME, encoding='utf-8', level=logging.DEBUG)

    # Build download database
    init_download_db()

    # Create download path if it doesn't exist
    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

    # Populate the database with a list of available PDFs, then download them
    with httpx.Client(timeout=120.0) as client:
        with closing(sqlite3.connect(DOWNLOAD_DB_PATH)) as conn:
            with conn:
                cursor = conn.cursor()

                # Get a list of reports with availabel PDFS, save document IDs, URLs, etc.
                get_metadata(client, cursor)

                # Loop through all PDFs once and attempt to download
                download_files(client, cursor)

                # Create chunks by combining document metadata with extracted text
                combined_chunks = extract_pdfs(cursor)

    # Build the index and save it, return the built index
    retriever = build_index(combined_chunks)

    # Run each test query through the retriever and print the top K results
    run_results = {
        'k': K,
        'queries': []
    }
    for query in TEST_QUERIES:
        logger.info('Running query: %s', query)
        print(f"Query: {query}\nResults:\n")

        query_results = []
        results, scores = doc_search_lexical(retriever, query, K)
        for i in range(results.shape[1]):
            result_chunk = results[0, i]
            query_results.append({
                'rank': i + 1,
                'score': float(scores[0, i]),
                'chunk_id': result_chunk['chunk_id'],
                'document_id': result_chunk['document_id'],
                'document_title': result_chunk['title'],
                'page_number': result_chunk['page_number']
            })

            print(f"Document Title: {result_chunk['title']}, Page {result_chunk['page_number']}")
            print(f"Text:\n{result_chunk['text']}\n\n")
            logger.info('Chunk Result: ID - %s, Score - %f, Document Title - %s, Page - %i',
                result_chunk['chunk_id'],
                scores[0, i],
                result_chunk['title'],
                result_chunk['page_number']
            )

        run_results['queries'].append({'query': query,
                            'results': query_results
        })

    # Print results to JSON file
    with open('run_results.json', mode='w', encoding='utf-8') as results_file:
        json.dump(run_results, results_file)

    logger.info('All PDFs examined')


if __name__ == '__main__':
    main()
import httpx
import json
from pathlib import Path
import pytest
import sqlite3

import bm25s


def create_database(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS articles_status (
        id INTEGER PRIMARY KEY,
        status TEXT NOT NULL,
        url TEXT NOT NULL,
        filename TEXT NOT NULL,
        title TEXT NOT NULL
        );
    '''
    )

    return


@pytest.fixture
def test_index() -> bm25s.BM25:
    test_retriever = bm25s.BM25.load(Path('ntrs_index'), load_corpus=True)
    return test_retriever


@pytest.fixture
def test_queries() -> list:
    return [
    ('Is the Morse-Lyapunov controller fuel optimal', 20250003112),
    ('What are the parameters of TIRS-2', 20210012997),
    ('What class receivers are DORIS satellites equipped with', 20230014981),
    ('What is a Luby transform', 20230007643),
    ('What is the best performing centrality measure of node clusters', 20240014692),
    ('Where is the DSS-25 station located', 20230010185),
    ('What is the role of polyvinyl alcohol in Bi electroplating', 20260002377)
]


@pytest.fixture
def mock_metadata_response():
    # Reading saved response body JSON from file
    response_path = Path(__file__).parent / 'fixtures' / 'ntrs_metadata_response.json'
    response_json = None

    with open(response_path, 'r', encoding='utf-8') as file:
        response_json = json.load(file)

    # Creating request handler and mock transport with body
    def handle_request(request: httpx.Request):
        # If headers are needed in the future, they can go here
        return httpx.Response(
            status_code=200,
            headers={'content-type': 'application/json'},
            json=response_json
        )

    mock_transport = httpx.MockTransport(handle_request)
    with httpx.Client(transport=mock_transport) as mock_client:
        yield mock_client


@pytest.fixture
def mock_database():
    mock_conn = sqlite3.connect(':memory:')
    create_database(mock_conn)
    cursor = mock_conn.cursor()

    yield cursor

    mock_conn.close()
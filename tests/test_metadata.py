import httpx
import sqlite3
import pytest

from src.nasa_ntrs_rag.app import get_metadata

def test_metadata_from_response(mock_metadata_response: httpx.Client, mock_database: sqlite3.Cursor):
    get_metadata(mock_metadata_response, mock_database)
    mock_database.execute('SELECT * FROM articles_status')
    entries = mock_database.fetchall()

    assert len(entries) == 11


def test_metadata_ingestion_status(mock_database: sqlite3.Cursor):
    mock_database.execute('SELECT * FROM articles_status;')
    entries = mock_database.fetchall()

    for entry in entries:
        assert entry[1] == 'pending'
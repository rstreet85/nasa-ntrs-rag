import httpx
import sqlite3
import pytest

from nasa_ntrs_rag.app import get_metadata

def test_metadata_from_response(mock_metadata_response: httpx.Client, mock_database: sqlite3.Cursor):
    get_metadata(mock_metadata_response, mock_database)
    mock_database.execute('SELECT * FROM articles_status')
    entries = mock_database.fetchall()

    assert len(entries) == 11


def test_metadata_ingestion_status(mock_metadata_response: httpx.Client,mock_database: sqlite3.Cursor):
    get_metadata(mock_metadata_response, mock_database)
    mock_database.execute('SELECT id, status FROM articles_status;')
    entries = mock_database.fetchall()

    assert len(entries) == 11
    assert all(status == 'pending' for _, status in entries)
import httpx
import sqlite3
import pytest

from nasa_ntrs_rag.app import get_metadata

def test_metadata_from_response(mock_metadata_response: httpx.Client, mock_database: sqlite3.Cursor):
    '''Test that metadata ingestion works for the expected response and returns the expected number of results'''
    get_metadata(mock_metadata_response, mock_database)
    mock_database.execute('SELECT * FROM articles_status')
    entries = mock_database.fetchall()

    assert len(entries) == 11


def test_metadata_ingestion_status_on_first_ingestion(mock_metadata_response: httpx.Client,mock_database: sqlite3.Cursor):
    '''Test that metadata ingestion works for the expected response and returns the expected status after first update'''
    get_metadata(mock_metadata_response, mock_database)
    mock_database.execute('SELECT id, status FROM articles_status;')
    entries = mock_database.fetchall()

    assert len(entries) == 11
    assert all(status == 'pending' for _, status in entries)


def test_metadata_ingestion_idempotent(mock_metadata_response: httpx.Client, mock_database: sqlite3.Cursor):
    '''Test that metadata ingestion works idempotently by keeping expected status if called twice in a row'''
    get_metadata(mock_metadata_response, mock_database)
    get_metadata(mock_metadata_response, mock_database)
    mock_database.execute('SELECT * FROM articles_status')
    entries = mock_database.fetchall()

    assert len(entries) == 11
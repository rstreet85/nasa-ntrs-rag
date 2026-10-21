import httpx
import sqlite3
import pytest

from src.nasa_ntrs_rag.app import get_metadata

def test_extraction(mock_metadata_response: httpx.Client, mock_database: sqlite3.Cursor):
    cursor = mock_database
    get_metadata(mock_metadata_response, cursor)

    return
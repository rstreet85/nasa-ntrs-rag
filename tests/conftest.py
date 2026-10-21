import json
from pathlib import Path
import pytest

import bm25s

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
def test_response() -> dict:
    pass
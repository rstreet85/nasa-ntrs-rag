from pathlib import Path
import pytest

import bm25s

@pytest.fixture
def test_index() -> bm25s.BM25:
    test_retriever = bm25s.BM25.load(Path('ntrs_index'), load_corpus=True)
    return test_retriever

@pytest.fixture
def test_queries():
    return [
    'Is the Morse-Lyapunov controller fuel optimal',
    'What are the parameters of TIRS-2',
    'What class receivers are DORIS satellites equipped with',
    'What is a Luby transform',
    'What is the best performing centrality measure of node clusters',
    'Where is the DSS-25 station located',
    'What is the role of polyvinyl alcohol in Bi electroplating'
]
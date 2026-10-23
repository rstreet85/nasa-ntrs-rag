import pytest
from nasa_ntrs_rag.app import doc_search_lexical

TEST_QUERIES = [
    ('Is the Morse-Lyapunov controller fuel optimal', 20250003112),
    ('What are the parameters of TIRS-2', 20210012997),
    ('What class receivers are DORIS satellites equipped with', 20230014981),
    ('What is a Luby transform', 20230007643),
    ('What is the best performing centrality measure of node clusters', 20240014692),
    ('Where is the DSS-25 station located', 20230010185),
    ('What is the role of polyvinyl alcohol in Bi electroplating', 20260002377)
]

@pytest.mark.parametrize('query, document_id', TEST_QUERIES)
def test_lexical_search_k1(test_index, query, document_id):
    results, scores = doc_search_lexical(test_index, query, k=1)
    assert document_id == results[0, 0]['document_id']


@pytest.mark.parametrize('query, document_id', TEST_QUERIES)
def test_lexical_search_k3(test_index, query, document_id):
    results, scores = doc_search_lexical(test_index, query, k=3)

    documents_found = []
    for i in range(results.shape[1]):
        documents_found.append(results[0, i]['document_id'])

    assert document_id in documents_found


@pytest.mark.parametrize('query, document_id', TEST_QUERIES)
def test_lexical_search_k5(test_index, query, document_id):
    results, scores = doc_search_lexical(test_index, query, k=5)

    documents_found = []
    for i in range(results.shape[1]):
        documents_found.append(results[0, i]['document_id'])

    assert document_id in documents_found
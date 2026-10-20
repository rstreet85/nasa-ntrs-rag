# RAG-based search of NASA's Technical Reports using NTRS

Experimental retrieval augmented generation system built on NASA's NTRS collection. The application queries the NTRS API for relevant documents, tracks document downloads, and uses PyMuPDF4LLM for text extraction. 

# Usage


# Pipeline

## Ingestion Flow

### Data Sources
The application queries the NTRS pubsearch server for a list of downloadable articles matching a subject code, which is parsed and saved to a SQLite database. Then the application iterates through all undownloaded articles, downloads the file, and updates the article status. Currently only reports with subject code of 'Space Communications, Spacecraft Communications, Command and Tracking' are downloaded.

### Parsing/Chunking

The application iterates through each downloaded PDF, and uses PyMuPDF4LLM to extract the text into fixed size chunks, where each chunk represents a page.

### Embedding

## Query Flow

### Lexical Search

The application uses BM25 (specifically the bm25s library) for keyword search, and returns the top K best matches.

### User Query Embedding and Search

### Prompt Engineering

### Semantic Search

# Benchmarks
Recall@K results for a limited dataset of 11 PDFs with the subject code 'Space Communications, Spacecraft Communications, Command and Tracking'. 

## Document-level Recall@K with BM25

| Query | Relevant Document IDs | Recall@1 | Recall@3 | Recall@5 |
| :---- | :-------------------- | :------: | :------: | :------: |
| Is the Morse-Lyapunov controller fuel optimal | 20250003112| 1.0 | 1.0 | 1.0 |
| What are the parameters of TIRS-2 | 20210012997 | 1.0 | 1.0 | 1.0 |
| What class receivers are DORIS satellites equipped with | 20230014981 | 1.0 | 1.0 | 1.0 |
| What is a Luby transform | 20230007643 | 1.0 | 1.0 | 1.0 |
| What is the best performing centrality measure of node clusters | 20240014692 | 1.0 | 1.0 | 1.0 |
| Where is the DSS-25 station located | 20230010185 |  1.0 | 1.0 | 1.0 |
| What is the role of polyvinyl alcohol in Bi electroplating | 20260002377 | 1.0 | 1.0 | 1.0 |

## Chunk-level Recall@K with BM25

| Query | Relevant Chunk IDs | Recall@1 | Recall@3 | Recall@5 |
| :---- | :-------------------- | :------: | :------: | :------: |
| Is the Morse-Lyapunov controller fuel optimal | 20250003112-9| 0.0 | 1.0 | 1.0 |
| What are the parameters of TIRS-2 | 20210012997-0 | 1.0 | 1.0 | 1.0 |
| What class receivers are DORIS satellites equipped with | 20230014981-17 | 1.0 | 1.0 | 1.0 |
| What is a Luby transform | 20230007643-3, 20230007643-4, 20230007643-9 | 0.33 | 0.33 | 0.67 |
| What is the best performing centrality measure of node clusters | 20240014692-0, 20240014692-1 | 0.5 | 0.5 | 1.0 |
| Where is the DSS-25 station located | 20230010185-5, 20230010185-10 |  0.5 | 0.5 | 0.5 |
| What is the role of polyvinyl alcohol in Bi electroplating | 20260002377-1, 20260002377-2 | 0.5 | 1.0 | 1.0 |

# Current Status

| Feature                             | Status    |
| :---------------------------------- | :-------: |
| NTRS API integration                | Completed |
| Publication filtering               | Completed |
| Increase number of subjects         | Pending   |
| Tracked downloads, basic            | Completed |
| Multi-threaded downloads            | Pending   |
| PDF text extraction                 | Completed |
| Text chunking, basic                | Completed |
| Top K keyword retrieval, basic      | Completed |
| Recall@K benchmark for BM25 results | Completed |
| Text embedding                      | Pending   |
| Embedding-based retrieval, basic    | Pending   |
| User query embedding                | Pending   |
| Vector querying                     | Pending   |
| Prompt construction                 | Pending   |
| LLM response generation             | Pending   |
| Split into Indexing API/Query API   | Pending   |
| pytest - Basic retrieval            | Completed |
# RAG-based search of NASA's Technical Reports using NTRS

Experimental retrieval augmented generation system built on NASA's NTRS collection. The application queries the NTRS API for relevant documents, tracks document downloads, and uses PyMuPDF4LLM for text extraction. 

# Pipeline

## Ingestion Flow

### Data Sources
The application queries the NTRS pubsearch server for a list of downloadable articles matching a subject code, which is parsed and saved to a SQLite database. Then the application iterates through all undownloaded articles, downloads the file, and updates the article status.

### Parsing/Chunking

The application iterates through each downloaded PDF, and uses PyMuPDF4LLM to extract the text into chunks. 

### Embedding

## Query Flow

### Lexical Search

The application uses BM25 (specifically the bm25s library) for keyword search, and returns the top K best matches.

### User Query Embedding and Search

### Prompt Engineering

### Semantic Search

# Current Status

Ingestion Flow
| Feature                           | Status    |
| --------------------------------- | --------- |
| NTRS API integration              | Completed |
| Publication filtering             | Completed |
| Increase number of subjects       | Pending   |
| Tracked downloads, basic          | Completed |
| Multi-threaded downloads          | Pending   |
| PDF text extraction               | Completed |
| Text chunking, basic              | Completed |
| Top K keyword retrieval, basic    | Completed |
| Text embedding                    | Pending   |
| Embedding-based retrieval, basic  | Pending   |
| User query embedding              | Pending   |
| Vector querying                   | Pending   |
| Prompt construction               | Pending   |
| LLM response generation           | Pending   |
| Split into Indexing API/Query API | Pending   |
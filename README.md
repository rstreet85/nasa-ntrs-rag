# RAG-based search of NASA's Technical Reports using NTRS

Experimental retrieval augmented generation system built on NASA's NTRS collection. The application queries the NTRS API for relevant documents, tracks document downloads, and uses PyMuPDF4LLM for text extraction. 

## Pipeline

### Ingestion

The application queries the NTRS pubsearch server for a list of downloadable articles matching a subject code, which is parsed and saved to a SQLite database. Then the application iterates through all undownloaded articles, downloads the file, and updates the article status.

### Extraction and Chunking

Iterates through each downloaded PDF and uses PyMuPDF4LLM to extract the text into chunks. 

## Current Status
Completed:
- NTRS API integration
- Publication filtering
  - Space Communications, Spacecraft Communications, Command and Tracking
- SQLite database to track downloads
- PDF text extraction
- Page-level segmentation
- Test retrieval with BM25

Pending:
- Increase number of related subject categories
- Multi-threaded downloads
- Embedding-based retrieval
- LLM generation
- Evaluation
- UI
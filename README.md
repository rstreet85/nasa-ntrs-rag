# RAG-based search of NASA's Technical Reports using NTRS

First attempt at building a RAG system. The document corpus is NASA's Technical Reports Servers (NTRS).

## Pipeline

### Ingestion

The application queries the NTRS pubsearch server for a list of downloadable articles matching a subject code, which is parsed and saved to a SQLite database. Then the application iterates through all undownloaded articles, downloads the file, and updates the article status.
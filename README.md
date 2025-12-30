# Enterprise AI Ingestion & Knowledge Graph Platform

This project provides an end-to-end pipeline for ingesting, normalizing, and extracting knowledge from various enterprise data sources. It transforms structured, semi-structured, and unstructured data into a unified format suitable for RAG pipelines and Neo4j Knowledge Graph construction.

## 🚀 Features

- **Multi-Source Ingestion**: Automatically processes CSVs, text files, and PDFs from structured and unstructured directories.
- **Data Normalization**: Cleans and normalizes disparate datasets into a standard JSON schema.
- **Knowledge Graph Extraction**: Rule-based extraction of (Subject)-[Relation]->(Object) triples from unstructured text.
- **Neo4j Ready**: Generates CSV and JSON outputs optimized for graph database ingestion.

## 📂 Project Structure

```text
├── structured/             # CSV files (Employees, Products)
├── semi_structured/        # CRM Leads, IT Tickets CSVs
├── unstructured/           # Text policies, meeting notes, reports
├── output/                 # Processed JSON/CSV results
├── ingest.py               # Core ingestion and normalization engine
├── extract_triples.py      # Entity-Relationship extraction script
└── enterprise_schema.sql   # SQL schema for relational storage
```

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+

### Ingestion Pipeline
To ingest all data from the source directories into the normalized `output/` folder:
```bash
python ingest.py
```

### Knowledge Graph Extraction
To extract entities and relationships from the unstructured text for a Knowledge Graph:
```bash
python extract_triples.py
```
This generates:
- `output/knowledge_graph_triples.json`
- `output/knowledge_graph_triples.csv`

## 📊 Data Mapping (Triples)

The extracted knowledge graph includes relationships such as:
- `(Employee)-[:ATTENDED]->(Meeting)`
- `(Platform)-[:GENERATED_REVENUE]->(Value)`
- `(Client)-[:USES]->(Platform)`
- `(Company)-[:INVESTS_IN]->(Technology)`

## 📬 Contact
**Author**: Harikrishnan
**Email**: harikanjingattu@gmail.com
**Repository**: [info_pro](https://github.com/Harikrishnankanjingattu/info_pro)
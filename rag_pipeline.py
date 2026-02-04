import os
import json
import pandas as pd
import faiss
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
MODEL_NAME = 'all-MiniLM-L6-v2'

class RAGPipeline:
    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.documents = []
        self.graph = nx.DiGraph()
        self.chunk_size = 500
        self.chunk_overlap = 100

    def chunk_text(self, text):
        """Splits text into smaller chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def load_data(self):
        """Loads all processed data from the output directory."""
        print("Loading data for embedding...")
        
        # 1. Load Unstructured Data
        unstructured_file = os.path.join(OUTPUT_DIR, "unstructured_ingestion.json")
        if os.path.exists(unstructured_file):
            with open(unstructured_file, 'r') as f:
                data = json.load(f)
                for record in data:
                    content = record['attributes']['content']
                    source = record['source_name']
                    chunks = self.chunk_text(content)
                    for i, chunk in enumerate(chunks):
                        text = f"Source: {source} (Part {i+1}) | Content: {chunk}"
                        self.documents.append({"text": text, "metadata": record, "chunk_id": i})

        # 2. Load Structured & Semi-Structured Data
        # We iterate through all JSON files in output that aren't triples or unstructured_ingestion
        exclude_files = ["knowledge_graph_triples.json", "unstructured_ingestion.json", 
                         "triples_structured.json", "triples_unstructured.json", 
                         "triples_semi_structured.json"]
        
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith(".json") and filename not in exclude_files:
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    for record in data:
                        # Convert attributes to a descriptive string
                        attr_str = ", ".join([f"{k}: {v}" for k, v in record['attributes'].items()])
                        text = f"Entity: {record['entity_type']} ({record['entity_id']}) | Data: {attr_str}"
                        self.documents.append({"text": text, "metadata": record})

        # 3. Load Triples for the Knowledge Graph
        triples_file = os.path.join(OUTPUT_DIR, "knowledge_graph_triples.json")
        if os.path.exists(triples_file):
            with open(triples_file, 'r') as f:
                triples = json.load(f)
                for t in triples:
                    self.graph.add_edge(t['subject'], t['object'], relation=t['relation'])
                    # Also add triples as searchable text
                    text = f"{t['subject']} {t['relation']} {t['object']}"
                    self.documents.append({"text": text, "metadata": t})

        print(f"Total documents prepared for indexing: {len(self.documents)}")

    def create_vector_db(self):
        """Generates embeddings and stores them in a FAISS index."""
        if not self.documents:
            print("No documents to index.")
            return

        print("Generating embeddings (this may take a moment)...")
        texts = [doc['text'] for doc in self.documents]
        embeddings = self.model.encode(texts)
        
        # Convert to float32 for FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Save index
        faiss.write_index(self.index, os.path.join(OUTPUT_DIR, "vector_db.index"))
        print(f"FAISS index created with {self.index.ntotal} vectors.")

    def synthesize_answer(self, query_text, segments, threshold=0.5):
        """Synthesizes a high-accuracy response with thresholding and keyword verification."""
        if not segments:
            return "I'm sorry, I couldn't find any information related to your request."

        # Filter by stricter relevance threshold
        valid_segments = [s for s in segments if (1 / (1 + s['distance'])) >= threshold]
        
        # Keyword verification (Simple check for high-confidence entities like locations or names)
        # This helps prevent a 'Mumbai' employee appearing for a 'Chennai' query.
        verified_segments = []
        query_words = set(query_text.lower().split())
        
        for s in valid_segments:
            searchable_text = s['doc']['text'].lower()
            # If the query specifies a location or specific entity, it MUST be in the segment
            # We look for words that look like Proper Nouns in the query
            proper_nouns = [w for w in query_text.split() if w[0].isupper() or w.lower() in ['chennai', 'mumbai', 'bengaluru', 'pune', 'noida', 'hyderabad', 'gurugram']]
            
            if proper_nouns:
                match = any(pn.lower() in searchable_text for pn in proper_nouns)
                if match:
                    verified_segments.append(s)
            else:
                verified_segments.append(s)

        if not verified_segments:
            return "\033[1;31mNO HIGH-ACCURACY MATCH FOUND.\033[0m\nI found some distant records, but none specifically matching your criteria (e.g., location/name). To avoid hallucination, I have suppressed those results."

        sources = set()
        structured_info = []
        facts = []
        document_snippets = []

        for s in verified_segments:
            doc = s['doc']
            meta = doc.get('metadata', {})
            sources.add(meta.get('source_name', 'Internal Records'))

            if "attributes" in meta:
                attrs = meta['attributes']
                # Create a specific fact string based on entity type
                if meta.get('entity_type') == 'Employee':
                    info = f"Employee {attrs.get('full_name')} ({meta.get('entity_id')}) is a {attrs.get('designation')} in {attrs.get('department')} ({attrs.get('location')})."
                elif meta.get('entity_type') == 'Client':
                    info = f"Client {attrs.get('client_name')} is in {attrs.get('industry')} (Country: {attrs.get('country')})."
                else:
                    attr_list = [f"{k}: {v}" for k, v in attrs.items() if k not in ['ingestion_timestamp', 'origin_system']]
                    info = f"{meta.get('entity_type')}: " + ", ".join(attr_list)
                
                if info not in structured_info:
                    structured_info.append(info)
            elif 'subject' in doc['metadata']:
                t = doc['metadata']
                facts.append(f"• {t['subject']} {t['relation']} {t['object']}")
            else:
                document_snippets.append(doc['text'].split('| Content: ')[-1])

        # Calculate average accuracy percentage for verified segments
        avg_dist = sum(s['distance'] for s in verified_segments) / len(verified_segments)
        # Convert distance to accuracy: 1.0 distance -> 0%, 0.0 distance -> 100% (approx)
        # Using a simple 1 / (1 + dist) for accuracy, but let's make it more "human" for %
        # A distance of 0.5 is usually very good, 1.0 is okay.
        # Let's use max(0, 100 * (1 - avg_dist)) as a heuristic for "Accuracy"
        accuracy_pct = max(0, min(100, (1 / (1 + avg_dist)) * 100))
        
        # Build Response
        response = f"\033[1;32mRETRIEVAL ACCURACY: {accuracy_pct:.1f}%\033[0m\n"
        response += "\033[1; RESPONSE :\033[0m\n"
        
        if structured_info:
            for info in structured_info:
                response += f"✓ {info}\n"
        
        if facts:
            for fact in facts:
                response += f"  {fact}\n"

        if document_snippets:
            for snippet in document_snippets[:1]: # Only the very best snippet
                response += f"\n\033[1;33mReference Documentation:\033[0m\n  > \"{snippet[:300]}...\"\n"

        response += f"\n\n\033[1;30mSource Files: {', '.join([f'[[{s}]]' for s in sources])}\033[0m"
        
        return response

    def query(self, query_text, top_k=5):
        """Retrieves relevant context and synthesizes a response."""
        if self.index is None:
            return "Index not initialized."

        query_vector = self.model.encode([query_text])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        retrieved_segments = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx < len(self.documents):
                retrieved_segments.append({
                    "doc": self.documents[idx],
                    "distance": distances[0][i]
                })
        
        ans = self.synthesize_answer(query_text, retrieved_segments)
        return ans


    def visualize_graph(self):
        """Saves a visualization of the Knowledge Graph."""
        if len(self.graph) == 0:
            print("Knowledge graph is empty.")
            return
            
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', 
                node_size=2000, edge_color='gray', font_size=10, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        plt.title("Dynamic Knowledge Graph")
        save_path = os.path.join(OUTPUT_DIR, "knowledge_graph.png")
        plt.savefig(save_path)
        print(f"Knowledge Graph visualization saved to {save_path}")

if __name__ == "__main__":
    rag = RAGPipeline()
    rag.load_data()
    rag.create_vector_db()
    
    # Example Queries
    rag.query("Who attended the Enterprise AI Strategy Review?")
    rag.query("What are the company policies regarding security?")
    rag.query("Capgemini have total revenue")

    
    rag.visualize_graph()

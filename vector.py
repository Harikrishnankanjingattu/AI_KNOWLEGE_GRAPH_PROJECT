import faiss
import os
import numpy as np

index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "vector_db.index")

if os.path.exists(index_path):
    # Load the binary index
    index = faiss.read_index(index_path)
    
    print(f"✅ Vector Database Loaded Successfully")
    print(f"🔹 Total Vectors (Items) stored: {index.ntotal}")
    print(f"🔹 Vector Dimension: {index.d}")

    # To see the actual vectors (the raw numbers):
    print("\n--- Displaying the first 2 vectors (raw numbers) ---")
    
    # We use reconstruct_n to get the actual float arrays stored in the index
    # (Note: This works only for Flat indexes like the one we created)
    raw_vectors = index.reconstruct_n(0, 2) 
    
    for i, vec in enumerate(raw_vectors):
        print(f"\nVector {i} (Showing first 10 of {index.d} numbers):")
        # These are the embeddings that the model generated
        print(vec[:10]) 
        print("...")
else:
    print("❌ Index file not found!")
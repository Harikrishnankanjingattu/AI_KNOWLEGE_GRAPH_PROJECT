import os
import json
import csv
import datetime
import uuid
import hashlib

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
STRUCTURED_DIR = os.path.join(DATA_DIR, "structured")
SEMI_STRUCTURED_DIR = os.path.join(DATA_DIR, "semi_structured")
UNSTRUCTURED_DIR = os.path.join(DATA_DIR, "unstructured")

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def generate_id(prefix, *args):
    """Generates a deterministic ID based on input arguments."""
    content = "-".join(map(str, args))
    hash_object = hashlib.md5(content.encode())
    return f"{prefix}-{hash_object.hexdigest()[:8]}"

def get_iso_timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def normalize_record(source_type, source_name, entity_type, attributes, entity_id=None, relationships=None):
    if entity_id is None:
     
        entity_id = generate_id(entity_type[:3].upper(), json.dumps(attributes, sort_keys=True))
    
    if relationships is None:
        relationships = []

    if "department" in attributes:
        relationships.append({
            "relation": "BELONGS_TO",
            "target_entity": generate_id("DEP", attributes["department"])
        })
 
    if entity_type == "Employee":
         if "email" in attributes:
        
             attributes["email"] = attributes["email"].lower()

    return {
        "source_type": source_type,
        "source_name": source_name,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "attributes": attributes,
        "relationships": relationships,
        "metadata": {
            "ingestion_timestamp": get_iso_timestamp(),
            "source_confidence": 1.0 if source_type == "structured" else 0.8,
            "origin_system": "File System"
        }
    }

def process_structured():
    print("Processing Structured Data...")
    if not os.path.exists(STRUCTURED_DIR):
        print(f"Directory not found: {STRUCTURED_DIR}")
        return

    for filename in os.listdir(STRUCTURED_DIR):
        if not filename.endswith(".csv"):
            continue
            
        filepath = os.path.join(STRUCTURED_DIR, filename)
        entity_type = filename.replace(".csv", "").rstrip('s').title() # employees -> Employee
        
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
           
                entity_id = None
                if "employee_id" in row:
                    entity_id = row["employee_id"]
                elif "product_id" in row:
                     entity_id = row["product_id"]
                
                normalized = normalize_record(
                    source_type="structured",
                    source_name=filename,
                    entity_type=entity_type,
                    attributes=row,
                    entity_id=entity_id
                )
                records.append(normalized)
        
        output_file = os.path.join(OUTPUT_DIR, f"{filename.replace('.csv', '')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)
        print(f"Processed {len(records)} records from {filename}")

def process_semi_structured():
    print("Processing Semi-Structured Data...")
    if not os.path.exists(SEMI_STRUCTURED_DIR):
         print(f"Directory not found: {SEMI_STRUCTURED_DIR}")
         return

    for filename in os.listdir(SEMI_STRUCTURED_DIR):
        if not filename.endswith(".csv"):
            continue
            
        filepath = os.path.join(SEMI_STRUCTURED_DIR, filename)
        # crm_leads -> Lead, it_tickets -> Issue
        entity_type = "Entity"
        if "lead" in filename: entity_type = "Lead"
        elif "ticket" in filename: entity_type = "Issue"
        elif "vendor" in filename: entity_type = "Vendor"
        
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entity_id = None
                if "ticket_id" in row: entity_id = row["ticket_id"]
                if "lead_id" in row: entity_id = row["lead_id"]
                
                normalized = normalize_record(
                    source_type="semi_structured",
                    source_name=filename,
                    entity_type=entity_type,
                    attributes=row,
                    entity_id=entity_id
                )
                records.append(normalized)

        output_file = os.path.join(OUTPUT_DIR, f"{filename.replace('.csv', '')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)
        print(f"Processed {len(records)} records from {filename}")

def process_unstructured():
    print("Processing Unstructured Data...")
    if not os.path.exists(UNSTRUCTURED_DIR):
         print(f"Directory not found: {UNSTRUCTURED_DIR}")
         return
         
    records = []
    for filename in os.listdir(UNSTRUCTURED_DIR):
        filepath = os.path.join(UNSTRUCTURED_DIR, filename)
        
        content = ""
        try:
            if filename.endswith(".txt"):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif filename.endswith(".pdf"):
            
                content = "[Binary PDF Content - Timestamp: " + get_iso_timestamp() + "]"
            else:
                continue
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

     
        entity_type = "Document"
        if "policy" in filename.lower(): entity_type = "Policy"
        if "report" in filename.lower(): entity_type = "Report"
        
      
        chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
        if not chunks:
            chunks = [content]
            
        for i, chunk in enumerate(chunks):
             normalized = normalize_record(
                source_type="unstructured",
                source_name=filename,
                entity_type=entity_type,
                attributes={"content": chunk, "chunk_index": i},
                entity_id=generate_id(entity_type[:3].upper(), filename, i)
            )
             records.append(normalized)

    output_file = os.path.join(OUTPUT_DIR, "unstructured_ingestion.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    print(f"Processed {len(records)} chunks from unstructured data")

def main():
    print("Starting Ingestion Engine...")
    ensure_output_dir()
    process_structured()
    process_semi_structured()
    process_unstructured()
    print("Ingestion Complete. Output saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()

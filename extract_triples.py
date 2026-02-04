import os
import json
import re
import csv

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
STRUCTURED_DIR = os.path.join(DATA_DIR, "structured")
SEMI_STRUCTURED_DIR = os.path.join(DATA_DIR, "semi_structured")
UNSTRUCTURED_DIR = os.path.join(DATA_DIR, "unstructured")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

def save_triples(triples, base_filename):
    """Helper function to save triples to JSON and CSV."""
    if not triples:
        return
        

    json_file = os.path.join(OUTPUT_DIR, f"{base_filename}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(triples, f, indent=2)

  
    csv_file = os.path.join(OUTPUT_DIR, f"{base_filename}.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("subject,relation,object,source\n")
        for t in triples:
            sub = str(t["subject"]).replace('"', '""')
            rel = str(t["relation"]).replace('"', '""')
            obj = str(t["object"]).replace('"', '""')
            src = str(t["source"]).replace('"', '""')
            f.write(f'"{sub}","{rel}","{obj}","{src}"\n')
    
    print(f"Saved {len(triples)} triples to {base_filename}.json/csv")

def extract_triples_from_text(filename, text):
    triples = []

    if "meeting_notes" in filename:
        attendees_match = re.search(r"Attendees:\s*(.*)", text)
        if attendees_match:
            attendees = [a.strip() for a in attendees_match.group(1).split(",")]
            meeting_name = "Enterprise AI Strategy Review"
            for attendee in attendees:
                triples.append({"subject": attendee, "relation": "ATTENDED", "object": meeting_name})
        
        discussion_match = re.search(r"Discussion:\s*(.*?)(?=Action Items:|$)", text, re.DOTALL)
        if discussion_match:
            topics = re.findall(r"- (.*)", discussion_match.group(1))
            for topic in topics:
                if "InsightAI" in topic and "healthcare diagnostics" in topic:
                    triples.append({"subject": "InsightAI", "relation": "EXPANDING_INTO", "object": "Healthcare Diagnostics"})
                if "Apollo Hospitals" in topic:
                    triples.append({"subject": "Apollo Hospitals", "relation": "DEMANDS", "object": "InsightAI"})
                if "Fortis Healthcare" in topic:
                    triples.append({"subject": "Fortis Healthcare", "relation": "DEMANDS", "object": "InsightAI"})

    elif "email_communications" in filename:
        sender_match = re.search(r"From:\s*(.*)", text)
        receiver_match = re.search(r"To:\s*(.*)", text)
        if sender_match and receiver_match:
            triples.append({"subject": sender_match.group(1).strip(), "relation": "SENT_EMAIL_TO", "object": receiver_match.group(1).strip()})
        
        revenue_match = re.search(r"InsightAI platform revenue crossed USD ([\d.]+) million", text)
        if revenue_match:
             triples.append({"subject": "InsightAI", "relation": "GENERATED_REVENUE", "object": f"USD {revenue_match.group(1)} Million"})
        
        clients_match = re.search(r"BFSI clients including (.*)\.", text)
        if clients_match:
            clients = [c.strip() for c in clients_match.group(1).split("and")]
            for client in clients:
                triples.append({"subject": client, "relation": "USES", "object": "InsightAI"})

    elif "company_policy" in filename:
        if "ISO 27001" in text:
            triples.append({"subject": "Employees", "relation": "COMPLIES_WITH", "object": "ISO 27001"})
        if "role-based access control" in text:
            triples.append({"subject": "Enterprise Data", "relation": "GOVERNED_BY", "object": "RBAC"})
        if "Information Security Office" in text:
            triples.append({"subject": "Data Breach", "relation": "REPORTED_TO", "object": "Information Security Office"})

    elif "industry_reports" in filename:
        if "Indian AI analytics market" in text:
            triples.append({"subject": "Indian AI Analytics Market", "relation": "GROWING_AT", "object": "25% CAGR"})
        players = ["TCS", "Infosys", "Wipro"]
        for player in players:
            if player in text:
                triples.append({"subject": player, "relation": "INVESTS_IN", "object": "Knowledge Graph Platforms"})

    elif "support_cases" in filename:
        client_match = re.search(r"Client:\s*(.*)", text)
        issue_match = re.search(r"Issue:\s*(.*)", text)
        resolution_match = re.search(r"Resolution:\s*(.*)", text)
        if client_match and issue_match:
            triples.append({"subject": client_match.group(1).strip(), "relation": "REPORTED_ISSUE", "object": issue_match.group(1).strip()})
        if issue_match and resolution_match:
            triples.append({"subject": issue_match.group(1).strip(), "relation": "RESOLVED_BY", "object": resolution_match.group(1).strip()})

    return triples

def extract_triples_from_structured(filename, filepath):
    triples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "employees" in filename:
                triples.append({"subject": row["full_name"], "relation": "WORKS_IN", "object": row["department"]})
                triples.append({"subject": row["full_name"], "relation": "HAS_DESIGNATION", "object": row["designation"]})
            elif "clients" in filename:
                triples.append({"subject": row["client_name"], "relation": "BELONGS_TO_INDUSTRY", "object": row["industry"]})
                triples.append({"subject": row["client_name"], "relation": "LOCATED_IN", "object": row["country"]})
            elif "products" in filename:
                triples.append({"subject": row["product_name"], "relation": "BELONGS_TO_CATEGORY", "object": row["category"]})
            elif "projects" in filename:
                triples.append({"subject": row["project_name"], "relation": "INVOLVES_PRODUCT", "object": row["product_name"]})
                triples.append({"subject": row["project_name"], "relation": "HAS_STATUS", "object": row["status"]})
            elif "company_revenue" in filename:
                triples.append({"subject": "Enterprise AI", "relation": "GENERATED_REVENUE", "object": f"{row['revenue_usd']} in {row['year']}"})
    return triples

def extract_triples_from_semi_structured(filename, filepath):
    triples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "assets" in filename:
                triples.append({"subject": row["asset_id"], "relation": "IS_A", "object": row["asset_type"]})
                triples.append({"subject": row["asset_id"], "relation": "LOCATED_AT", "object": row["location"]})
            elif "crm_leads" in filename:
                triples.append({"subject": row["company_name"], "relation": "INTERESTED_IN", "object": row["interested_product"]})
                triples.append({"subject": row["company_name"], "relation": "MANAGED_BY", "object": row["account_manager"]})
            elif "it_tickets" in filename:
                triples.append({"subject": row["ticket_id"], "relation": "RELATES_TO", "object": row["product_name"]})
                triples.append({"subject": row["ticket_id"], "relation": "HAS_STATUS", "object": row["status"]})
            elif "vendors" in filename:
                triples.append({"subject": row["vendor_name"], "relation": "PROVIDES_SERVICE", "object": row["service"]})
            elif "training_records" in filename:
                triples.append({"subject": row["training_id"], "relation": "FOR_COURSE", "object": row["course_name"]})
                triples.append({"subject": row["training_id"], "relation": "DEPARTMENT", "object": row["department"]})
    return triples

def main():
    unstructured_triples = []
    structured_triples = []
    semi_structured_triples = []
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

 
    if os.path.exists(UNSTRUCTURED_DIR):
        for filename in os.listdir(UNSTRUCTURED_DIR):
            if filename.endswith(".txt"):
                filepath = os.path.join(UNSTRUCTURED_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    triples = extract_triples_from_text(filename, content)
                    for t in triples:
                        t["source"] = f"unstructured/{filename}"
                    unstructured_triples.extend(triples)
    if os.path.exists(STRUCTURED_DIR):
        for filename in os.listdir(STRUCTURED_DIR):
            if filename.endswith(".csv"):
                filepath = os.path.join(STRUCTURED_DIR, filename)
                triples = extract_triples_from_structured(filename, filepath)
                for t in triples:
                    t["source"] = f"structured/{filename}"
                structured_triples.extend(triples)

    if os.path.exists(SEMI_STRUCTURED_DIR):
        for filename in os.listdir(SEMI_STRUCTURED_DIR):
            if filename.endswith(".csv"):
                filepath = os.path.join(SEMI_STRUCTURED_DIR, filename)
                triples = extract_triples_from_semi_structured(filename, filepath)
                for t in triples:
                    t["source"] = f"semi_structured/{filename}"
                semi_structured_triples.extend(triples)

    save_triples(unstructured_triples, "triples_unstructured")
    save_triples(structured_triples, "triples_structured")
    save_triples(semi_structured_triples, "triples_semi_structured")


    all_triples = unstructured_triples + structured_triples + semi_structured_triples
    save_triples(all_triples, "knowledge_graph_triples")

    print("-" * 30)
    print(f"Total Triples Extracted: {len(all_triples)}")

if __name__ == "__main__":
    main()

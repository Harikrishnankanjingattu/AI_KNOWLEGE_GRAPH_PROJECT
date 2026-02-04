import sys
import os
import time
from rag_pipeline import RAGPipeline

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\033[1;36m" + "="*60)
    print("      🚀 INSIGHT AI - INTELLIGENT RAG TERMINAL 🚀")
    print("="*60 + "\033[0m")
    
    print("\n[1/3] \033[1;33mInitializing RAG Pipeline...\033[0m")
    pipeline = RAGPipeline()
    
    print("[2/3] \033[1;33mLoading project data...\033[0m")
    pipeline.load_data()
    
    print("[3/3] \033[1;33mBuilding vector index for high-accuracy retrieval...\033[0m")
    pipeline.create_vector_db()

    print("\n\033[1;32mREADY! You can now ask questions about your enterprise data.\033[0m")
    print("\033[1;30m(Ask about employees, clients, IT tickets, revenue, or company policies)\033[0m")
    print("Type \033[1;31m'exit'\033[0m to close the session.\n")

    while True:
        try:
            query = input("\033[1;34m🤔 ASK ME ANYTING:\033[0m ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit']:
                print("\n\033[1;35mExiting... Have a great day!\033[0m")
                break
            
            print("\033[1;30m🔍 Searching enterprise intelligence...\033[0m", end="\r")
            
            print("\n" + "\033[1;34m" + "─"*60 + "\033[0m")
            pipeline.query(query)
            print("\033[1;34m" + "─"*60 + "\033[0m\n")
            
        except KeyboardInterrupt:
            print("\n\n\033[1;35mGoodbye!\033[0m")
            break
        except Exception as e:
            print(f"\n\033[1;31mAn error occurred: {e}\033[0m")

if __name__ == "__main__":
    main()

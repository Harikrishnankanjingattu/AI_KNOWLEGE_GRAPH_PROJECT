from flask import Flask, request, jsonify, render_template
import os
from rag_pipeline import RAGPipeline

app = Flask(__name__)


pipeline = RAGPipeline()
pipeline.load_data()
pipeline.create_vector_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({"answer": "Please provide a query."}), 400
    
    try:
        answer = pipeline.query(user_query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
   
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app.run(debug=True, port=5000)

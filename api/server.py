import os
import spacy
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load spaCy model from project directory
model_path = os.path.join(os.path.dirname(__file__), '..', 'spacy_model', 'en_core_web_sm', 'en_core_web_sm-3.8.0')
nlp = spacy.load(model_path)


@app.route('/')
def serve_index():
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
    static_path = os.path.join(static_dir, 'index.html')
    print(f"Checking for index.html at: {static_path}")
    if not os.path.exists(static_path):
        print(f"File not found at: {static_path}")
        return "index.html not found", 404
    print(f"Serving index.html from: {static_path}")
    return send_from_directory(static_dir, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
    static_path = os.path.join(static_dir, path)
    print(f"Checking for static file at: {static_path}")
    if os.path.exists(static_path):
        print(f"Serving static file from: {static_path}")
        return send_from_directory(static_dir, path)
    print(f"File not found at: {static_path}")
    return "File not found", 404

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Global variables to store document data
sentences = []
sentence_embeddings = []

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return doc

def get_embedding(doc):
    vectors = [token.vector for token in doc if not token.is_stop and not token.is_punct]
    if not vectors:
        return np.zeros(nlp.vocab.vectors.shape[1])
    return np.mean(vectors, axis=0)

def process_document(document):
    doc = nlp(document)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    sentence_docs = [preprocess_text(sent) for sent in sentences]
    sentence_embeddings = [get_embedding(doc) for doc in sentence_docs]
    return sentences, sentence_embeddings

def get_best_response(user_input, sentence_embeddings, sentences):
    input_doc = preprocess_text(user_input)
    input_embedding = get_embedding(input_doc)
    similarities = cosine_similarity([input_embedding], sentence_embeddings)[0]
    threshold = 0.7
    top_indices = np.argsort(similarities)[::-1][:4]
    responses = []
    for idx in top_indices:
        if similarities[idx] > threshold:
            responses.append((sentences[idx], float(similarities[idx])))
        else:
            break
    if not responses:
        return "Sorry, I don’t have relevant information about that in the document.", 0.0
    return responses, float(max(similarities))

@app.route('/api/upload', methods=['POST'])
def upload_document():
    global sentences, sentence_embeddings
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    if file and file.filename.endswith('.txt'):
        try:
            document = file.read().decode('utf-8')
            sentences, sentence_embeddings = process_document(document)
            return jsonify({'success': True, 'message': 'Document processed successfully'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        return jsonify({'success': False, 'error': 'Invalid file format. Please upload a .txt file'}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    global sentences, sentence_embeddings
    data = request.get_json()
    query = data.get('query', '')
    if not sentences or not sentence_embeddings:
        return jsonify({'error': 'No document uploaded'}), 400
    if query.lower() in ['exit', 'quit']:
        return jsonify({'responses': 'Goodbye!'})
    response, max_similarity = get_best_response(query, sentence_embeddings, sentences)
    return jsonify({'responses': response, 'similarity': max_similarity})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
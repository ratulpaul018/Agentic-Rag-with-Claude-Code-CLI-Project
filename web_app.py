from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from werkzeug.utils import secure_filename
from book_qa import create_rag_chain, load_vector_store, create_vector_store, load_and_chunk_book

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global QA chain
qa_chain = None
vector_store_loaded = False

ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    global vector_store_loaded
    return render_template('index.html', vector_store_exists=vector_store_loaded)


@app.route('/api/upload', methods=['POST'])
def upload_book():
    global qa_chain, vector_store_loaded

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Only PDF files are allowed'}), 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the document
        chunks = load_and_chunk_book(filepath)

        # Rebuild vector store with ALL documents
        # Get all PDF files in uploads folder
        all_chunks = []
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            if file.endswith('.pdf'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                print(f"Loading {file} for vector store...")
                file_chunks = load_and_chunk_book(file_path)
                all_chunks.extend(file_chunks)
                print(f"Added {len(file_chunks)} chunks from {file}")

        # Create single vector store with ALL documents
        if all_chunks:
            vector_store = create_vector_store(all_chunks)
            print(f"Created vector store with {len(all_chunks)} total chunks from all documents")
        else:
            vector_store = create_vector_store(chunks)

        qa_chain = create_rag_chain(vector_store)
        vector_store_loaded = True

        return jsonify({
            'success': True,
            'message': f'Successfully processed {filename}',
            'chunks': len(chunks),
            'filename': filename
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question():
    global qa_chain

    try:
        if qa_chain is None:
            return jsonify({
                'success': False,
                'error': 'No book loaded. Please upload a PDF first.'
            }), 400

        data = request.json
        question = data.get('question', '').strip()

        if not question:
            return jsonify({'success': False, 'error': 'Question cannot be empty'}), 400

        # Get answer
        result = qa_chain.invoke({"query": question})

        # Format source documents
        sources = []
        if 'source_documents' in result:
            for doc in result['source_documents']:
                sources.append({
                    'page': doc.metadata.get('page', 'N/A'),
                    'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                })

        return jsonify({
            'success': True,
            'answer': result['result'],
            'sources': sources
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    global vector_store_loaded
    return jsonify({
        'vector_store_loaded': vector_store_loaded,
        'qa_chain_ready': qa_chain is not None
    })


@app.route('/api/get-pdf/<filename>')
def get_pdf(filename):
    """Serve uploaded PDF files"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path) and filename.endswith('.pdf'):
            return send_file(file_path, mimetype='application/pdf')
        return {"error": "File not found"}, 404
    except Exception as e:
        return {"error": str(e)}, 500


@app.route('/api/reset', methods=['POST'])
def reset():
    global qa_chain, vector_store_loaded
    try:
        qa_chain = None
        vector_store_loaded = False
        # Clear uploads
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        return jsonify({'success': True, 'message': 'System reset'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/load-existing', methods=['POST'])
def load_existing():
    """Load existing vector store"""
    global qa_chain, vector_store_loaded
    try:
        vector_store = load_vector_store()
        qa_chain = create_rag_chain(vector_store)
        vector_store_loaded = True
        return jsonify({'success': True, 'message': 'Loaded existing vector store'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

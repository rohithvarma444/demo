from flask import Flask, request, send_file, render_template_string
import os
import cohere
import chromadb
from pypdf import PdfReader
from datetime import datetime

# -------------------- CONFIGURATION --------------------
COHERE_API_KEY = "rV9UT11Rh4fMDVZrQzwCRwIeGCy6ywrXhXV0BHBt"
CHROMA_DB_PATH = "./chroma_db"
UPLOAD_FOLDER = "./uploads"
PDF_NAME = "SRS.pdf"  # Fixed PDF name

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Cohere client
cohere_client = cohere.Client(COHERE_API_KEY)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="documents")


# -------------------- FUNCTIONS --------------------

def extract_text_from_pdf(pdf_path):
    """ Extracts text from PDF """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""  # Handle pages with no text
    return text.strip()


def store_text_in_chroma(text):
    """ Stores PDF text as multilingual embeddings in ChromaDB """
    chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  # Unique timestamp

    for i, chunk in enumerate(chunks):
        embedding = cohere_client.embed(
            texts=[chunk],
            model="embed-multilingual-v3.0",  # Multilingual model
            input_type="search_document"
        ).embeddings[0]

        # Store with unique ID
        collection.add(
            ids=[f"SRS_{timestamp}_{i}"],
            embeddings=[embedding],
            metadatas=[{"content": chunk, "pdf": PDF_NAME}]
        )


def retrieve_relevant_text(query):
    """ Retrieves relevant multilingual text from ChromaDB """
    query_embedding = cohere_client.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    ).embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    if 'metadatas' not in results or not results['metadatas']:
        return ["No relevant content found."]

    return [meta['content'] for meta in results['metadatas'][0]]


def generate_response(query, context):
    """ Generates a response using Cohere """
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = cohere_client.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=300
    )

    return response.generations[0].text.strip()


# -------------------- ROUTES --------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    response = ""
    pdf_content = ""

    # Process the SRS.pdf file by default
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], PDF_NAME)

    if not os.path.exists(pdf_path):
        return "SRS.pdf not found in the uploads directory."

    # Store SRS.pdf content in ChromaDB
    document_text = extract_text_from_pdf(pdf_path)
    pdf_content = document_text[:2000] + "..." if len(document_text) > 2000 else document_text
    store_text_in_chroma(document_text)

    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            relevant_text = retrieve_relevant_text(query)
            context = " ".join(relevant_text)
            response = generate_response(query, context)

    # Example questions in multiple languages
    example_questions = [
        ("Telugu", "ఈ డాక్యుమెంట్‌లో ప్రాజెక్ట్ లక్ష్యాలు ఏమిటి?"),
        ("English", "What are the project objectives mentioned in this document?"),
        ("Hindi", "इस दस्तावेज़ में परियोजना के उद्देश्य क्या हैं?"),
        ("Malayalam", "ഈ പ്രോജക്ട് ഡോക്യുമെന്റിലെ ഉദ്ദേശ്യങ്ങൾ എന്തൊക്കെയാണു?")
    ]

    # HTML template
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>PDF Q&A System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f4f4f4;
                color: #333;
                margin: 0;
                padding: 0;
                line-height: 1.6;
            }
            .container {
                max-width: 1100px;
                margin: 30px auto;
                padding: 20px;
                background: white;
                box-shadow: 0 0 15px #ccc;
                border-radius: 10px;
            }
            h1, h2, h3 {
                text-align: center;
            }
            form {
                text-align: center;
                margin-bottom: 20px;
            }
            input, button {
                padding: 10px 15px;
                font-size: 16px;
                margin: 5px;
                width: 70%;
            }
            button {
                background: #28a745;
                color: white;
                border: none;
                cursor: pointer;
            }
            button:hover {
                background: #218838;
            }
            pre, .pdf-content {
                background: #f9f9f9;
                padding: 15px;
                border: 1px solid #ccc;
                white-space: pre-wrap;
                word-wrap: break-word;
                max-height: 300px;
                overflow-y: auto;
            }
            .pdf-view {
                background: #fff;
                border: 1px solid #ccc;
                margin-top: 20px;
                padding: 15px;
                max-height: 400px;
                overflow-y: auto;
            }
            .footer {
                text-align: center;
                margin-top: 20px;
                color: #888;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            li {
                font-size: 16px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PDF Q&A System (Multilingual)</h1>
            
            <div class="footer">
                🚧 This system is still under development. Please ask questions based on the PDF content. 🚧
            </div>

            <form method="POST">
                <input type="text" name="query" placeholder="Ask a question..." required>
                <button type="submit">Ask</button>
            </form>

            <h2>Answer:</h2>
            <pre>{{ response }}</pre>

            <h2>PDF Content Preview:</h2>
            <div class="pdf-view">{{ pdf_content }}</div>

            <h2>Example Questions in Multiple Languages:</h2>
            <ul>
                {% for lang, question in example_questions %}
                    <li><b>{{ lang }}:</b> {{ question }}</li>
                {% endfor %}
            </ul>
            
            <div class="footer">
                <a href="/pdf">📄 View Full PDF</a>
            </div>
        </div>
    </body>
    </html>
    """, response=response, pdf_content=pdf_content, example_questions=example_questions)


@app.route('/pdf')
def view_pdf():
    """ Serve the SRS.pdf for viewing """
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], PDF_NAME)

    if os.path.exists(pdf_path):
        return send_file(pdf_path)
    else:
        return "SRS.pdf not found.", 404


# -------------------- RUN FLASK --------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)

from flask import Flask, request, send_file
import os
import cohere
import chromadb
from pypdf import PdfReader

# -------------------- CONFIGURATION --------------------
COHERE_API_KEY = "fp2QVcrS07SOx3j0M9YFc97uj0QAtETbtxObs7cA"
CHROMA_DB_PATH = "./chroma_db"
UPLOAD_FOLDER = "./"

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
    """ Stores PDF text as embeddings in ChromaDB """
    chunks = [text[i:i + 512] for i in range(0, len(text), 512)]

    for i, chunk in enumerate(chunks):
        embedding = cohere_client.embed(
            texts=[chunk],
            model="embed-multilingual-v3.0",
            input_type="search_document"
        ).embeddings[0]

        collection.add(
            ids=[f"doc_{i}"],
            embeddings=[embedding],
            metadatas=[{"content": chunk}]
        )


def retrieve_relevant_text(query):
    """ Retrieves relevant text from ChromaDB """
    query_embedding = cohere_client.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    ).embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return [result["content"] for result in results["metadatas"][0]]


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
    pdf_path = "SRS.pdf"

    if request.method == 'POST':
        # Handle PDF upload
        if 'pdf_file' in request.files:
            file = request.files['pdf_file']
            if file.filename != '':
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], "SRS.pdf")
                file.save(pdf_path)

                # Extract and store text in ChromaDB
                document_text = extract_text_from_pdf(pdf_path)
                store_text_in_chroma(document_text)

        # Handle query
        query = request.form.get('query')
        if query:
            relevant_text = retrieve_relevant_text(query)
            context = " ".join(relevant_text)
            response = generate_response(query, context)

    # HTML and CSS combined
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>PDF Q&A</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                background: #f4f4f4;
                color: #333;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 900px;
                margin: 30px auto;
                padding: 20px;
                background: white;
                box-shadow: 0 0 10px #ccc;
                border-radius: 8px;
                text-align: center;
            }}
            h1 {{
                color: #333;
            }}
            form {{
                margin-bottom: 20px;
            }}
            input, button {{
                padding: 10px;
                font-size: 16px;
                margin: 5px;
            }}
            button {{
                background: #28a745;
                color: white;
                border: none;
                cursor: pointer;
            }}
            button:hover {{
                background: #218838;
            }}
            a {{
                color: #007bff;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            pre {{
                background: #f4f4f4;
                padding: 15px;
                border: 1px solid #ccc;
                white-space: pre-wrap;
                word-wrap: break-word;
                max-height: 300px;
                overflow-y: auto;
                text-align: left;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PDF Q&A System</h1>

            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="pdf_file" accept=".pdf" required>
                <button type="submit">Upload and Process PDF</button>
            </form>

            <form method="POST">
                <input type="text" name="query" placeholder="Ask a question..." required>
                <button type="submit">Ask</button>
            </form>

            <a href="/pdf">📄 View PDF</a>

            <h2>Answer:</h2>
            <pre>{response}</pre>
        </div>
    </body>
    </html>
    """


@app.route('/pdf')
def view_pdf():
    """ Serve the PDF for viewing """
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], "SRS.pdf")
    return send_file(pdf_path)


# -------------------- RUN FLASK --------------------
# For production, bind to 0.0.0.0 and use environment variable PORT or fallback to 10000
if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=PORT, debug=True)

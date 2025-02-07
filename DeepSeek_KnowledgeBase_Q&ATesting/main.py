import os
import PyPDF2
from docx import Document
import requests
import re
import logging
import pymongo  # MongoDB client for database operations
import random
import numpy as np
import uuid
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Jinja2 Templates
templates = Jinja2Templates(directory="templates")

# MongoDB connection
client = pymongo.MongoClient(
    "mongodb+srv://aremin963:PdBTqm5SpnK6Zdp9@resybotcluster.pqcri.mongodb.net/?retryWrites=true&w=majority&appName=ResybotCluster", 
    connect=False)  # Connect to MongoDB cluster; replace with your MongoDB URI
db = client["BBS_HR_AI_AGENT"]  # Specify database name
COLLECTION_NAME = "KnowledgeBase"

try:
    client.server_info()  # Test connection
    knowledge_collection = db[COLLECTION_NAME]
    logger.info("Successfully connected to MongoDB.")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise HTTPException(status_code=500, detail="Database connection error.")

# Directory for uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''.join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

# Extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text

def split_into_sentences(text):
    # Use NLTK's sentence tokenizer for accurate splitting
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def split_into_chunks(text, max_chunk_size=500):
    # Split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chunk_size:
            current_chunk += paragraph + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

# Query local AI model
def query_local_model(prompt, context):
    # Normalize the prompt for easier matching
    normalized_prompt = prompt.strip().lower()

    # Handle gratitude expressions
    gratitude_keywords = ["thank","thanks", "thank you", "tysm", "thank you so much", "appreciate"]
    gratitude_responses = [
        "You're very welcome! 😊 If you have any more questions, feel free to ask.",
        "Glad I could help! Let me know if there's anything else.💫",
        "No problem at all! 😄 I'm here if you need further assistance.",
        "My pleasure! Don't hesitate to reach out if you need more help.✨"
    ]
    if any(keyword in normalized_prompt for keyword in gratitude_keywords):
        return random.choice(gratitude_responses)
    
    # Handle generic or unclear inputs
    if normalized_prompt in ["hi", "hello", "hey"]:
        return "Hello! How may I assist you today? Please specify your question clearly."
    elif normalized_prompt in ["i need help", "help", "i need help!"]:
        return "Of course! I'd be happy to help. Could you please specify your question or the topic you need assistance with?"
    elif not normalized_prompt or len(normalized_prompt.split()) < 3:  # Very short or unclear input
        return "It seems your question is unclear. Could you please provide more details or specify your issues?"

    # Query the local model for relevant responses
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    # Updated prompt with strict instructions
    data = {
        "model": "deepseek-r1:8b",
        "prompt": (
            f"You are an expert BBS HR AI assistant trained to answer the following question using ONLY the information provided in the context below. "
            f"Respond in a clear, complete, and accurate point-form format. Prioritize specific details from the most relevant sections of the context, "
            f"but also consider the broader context if necessary. Do NOT provide any information outside the context. "
            f"If the answer is not found in the context, respond with 'Sorry, I'm unable to answer the question. Ensure there are no grammatical errors 😊'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\nAnswer:"
        ),
        "stream": False,
        "max_tokens": 100000,
        "temperature": 0  # Ensure deterministic responses
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        response_data = response.json()
        raw_response = response_data.get("response", "").strip()
        return clean_response(raw_response)
    else:
        raise HTTPException(status_code=500, detail=f"Error querying local model: {response.text}")

# Clean AI model response
def clean_response(response):
    # Remove unwanted tags and formatting
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"```[\s\S]*?```", "", response)  # Remove code blocks
    response = re.sub(r"\n{3,}", "\n\n", response).strip()  # Reduce excessive newlines
    
    # Ensure bold (**text**) and italic (_text_) are preserved
    response = re.sub(r"\*\*(.*?)\*\*", r"**\1**", response)  # Preserve bold formatting
    response = re.sub(r"_(.*?)_", r"_\1_", response)  # Preserve italic formatting
    
    # Handle incomplete sentences
    if response.endswith(("...", "Dete", "calcu")):
        response += "\n(Note: The response may be incomplete due to length restrictions.)"
    
    return response

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for knowledge bases
def generate_embeddings(text_chunks):
    return embedding_model.encode(text_chunks)

# Find the most relevant knowledge base
def find_relevant_knowledge_base(question, filenames):
    question_embedding = generate_embeddings([question])[0]
    file_embeddings = [generate_embeddings([doc["text"]]) for doc in knowledge_collection.find({"filename": {"$in": filenames}}, {"_id": 0, "text": 1})]
    similarities = [cosine_similarity([question_embedding], [emb])[0][0] for emb in file_embeddings]
    most_relevant_index = similarities.index(max(similarities))
    return filenames[most_relevant_index]



# Home route
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "Hello from FastAPI!"})

# Get uploaded files
@app.get("/files")
async def get_uploaded_files():
    try:
        logger.info("Fetching filenames from the database...")

        # Retrieve distinct filenames from MongoDB
        filenames = knowledge_collection.distinct("filename")

        logger.info(f"Fetched filenames: {filenames}")

        return {"files": filenames}

    except Exception as e:
        logger.error(f"Error fetching filenames: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching uploaded files.")



from pydantic import BaseModel
from typing import Optional

class FileMetadata(BaseModel):
    source: str
    category: str
    priority: int

@app.post("/upload")
async def upload_document(files: List[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {filename}")

        file_path = os.path.join(UPLOAD_FOLDER, filename)

        try:
            # Check if the file already exists in MongoDB
            existing_file = knowledge_collection.find_one({"filename": filename})
            if existing_file:
                logger.warning(f"File '{filename}' already exists in database. Skipping upload.")
                continue  # Skip inserting duplicate filenames

            # Save the file locally
            with open(file_path, "wb") as buffer:
                buffer.write(file.file.read())

            logger.info(f"File '{filename}' saved successfully.")

            # Extract text
            text = extract_text_from_pdf(file_path) if filename.endswith(".pdf") else extract_text_from_docx(file_path)

            # Split text into chunks
            chunks = split_into_chunks(text)

            # Generate embeddings
            chunk_embeddings = generate_embeddings(chunks)

            # Store chunks in MongoDB with a unique file ID
            file_id = str(uuid.uuid4())
            for i, chunk in enumerate(chunks):
                knowledge_collection.insert_one({
                    "file_id": file_id,
                    "filename": filename,
                    "chunk": chunk,
                    "embedding": chunk_embeddings[i].tolist()
                })

            uploaded_files.append(filename)
            logger.info(f"Stored '{filename}' in the database.")

        except Exception as e:
            logger.error(f"Error processing file '{filename}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file {filename}: {str(e)}")

    return {"message": "Documents uploaded successfully", "files": uploaded_files}

@app.post("/query")
async def query(data: dict):
    question = data.get("query")
    selected_file = data.get("selected_file")  # Optional: User can still manually select a file

    if not question:
        raise HTTPException(status_code=400, detail="No query provided")

    # Auto-detect the most relevant knowledge base if no file is selected
    if not selected_file:
        logger.info("No file selected. Auto-detecting the most relevant knowledge base...")
        try:
            # Fetch all filenames from the database
            filenames = list(knowledge_collection.distinct("filename"))
            if not filenames:
                return {"response": "No knowledge bases available to auto-detect."}

            # Generate embeddings for the question
            question_embedding = generate_embeddings([question])[0]

            # Compute similarity between the question and each file's content
            similarities = []
            for filename in filenames:
                # Retrieve all chunks for the file
                documents = list(knowledge_collection.find({"filename": filename}, {"_id": 0, "chunk": 1, "embedding": 1}))
                chunks = [doc["chunk"] for doc in documents]
                embeddings = np.array([doc["embedding"] for doc in documents])

                if embeddings.size == 0:
                    continue  # Skip files with no valid embeddings

                # Compute average embedding for the file
                file_embedding = np.mean(embeddings, axis=0)
                similarity = cosine_similarity([question_embedding], [file_embedding])[0][0]
                similarities.append((filename, similarity))

            # Find the most relevant file
            if not similarities:
                return {"response": "No valid embeddings found to auto-detect the knowledge base."}

            most_relevant_file, max_similarity = max(similarities, key=lambda x: x[1])
            logger.info(f"Most relevant file detected: {most_relevant_file} (Similarity: {max_similarity})")
            selected_file = most_relevant_file

            # Notify the user about the auto-detected file
            notification_message = f"I found the most relevant document for your question: '{selected_file}'.\n"
        except Exception as e:
            logger.error(f"Error during auto-detection: {str(e)}")
            return {"response": "An error occurred while auto-detecting the knowledge base."}
    else:
        notification_message = ""

    # Filter documents by the selected file (either auto-detected or manually selected)
    query_filter = {"filename": selected_file}
    documents = list(knowledge_collection.find(query_filter, {"_id": 0, "chunk": 1, "embedding": 1}))

    # Extract chunks and embeddings
    chunks = [doc.get("chunk", "") for doc in documents]
    embeddings = [doc.get("embedding", []) for doc in documents]

    if not chunks or not embeddings:
        return {"response": f"{notification_message}No relevant information found in the selected knowledge base."}

    # Ensure embeddings are valid before stacking
    embeddings = [np.array(emb) for emb in embeddings if len(emb) > 0]
    if not embeddings:
        return {"response": f"{notification_message}No valid embeddings found in the selected knowledge base."}

    try:
        embeddings = np.vstack(embeddings)
    except ValueError as e:
        logger.error(f"Error stacking embeddings: {str(e)}")
        return {"response": f"{notification_message}An error occurred while processing embeddings."}

    # Compute similarity between the question and chunks
    question_embedding = generate_embeddings([question])[0].reshape(1, -1)
    try:
        similarities = cosine_similarity(question_embedding, embeddings).flatten()
    except ValueError as e:
        logger.error(f"Error computing cosine similarity: {str(e)}")
        return {"response": f"{notification_message}An error occurred while processing the query."}

    # Get top 3 results
    top_indices = similarities.argsort()[-3:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]

    # Pass relevant chunks to AI model
    combined_context = "\n".join(relevant_chunks)
    ai_response = query_local_model(question, combined_context)

    # Combine notification message with AI response
    final_response = notification_message + ai_response
    return {"response": final_response}


# View a specific file's content
@app.get("/view/{filename}")
async def view_file(filename: str):
    # Find the file_id of the file
    document = knowledge_collection.find_one({"filename": filename}, {"_id": 0, "file_id": 1})
    if not document:
        logger.warning(f"File '{filename}' not found in database.")
        raise HTTPException(status_code=404, detail="File not found")

    file_id = document.get("file_id")

    # Retrieve all chunks for the file
    documents = knowledge_collection.find({"file_id": file_id}, {"_id": 0, "chunk": 1})
    chunks = [doc["chunk"] for doc in documents]

    if not chunks:
        logger.warning(f"Chunks for '{filename}' not found.")
        raise HTTPException(status_code=404, detail="No content found for this file.")

    return {"filename": filename, "content": "\n".join(chunks)}


# Delete a file from the database
@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    # Find the file_id of the file
    document = knowledge_collection.find_one({"filename": filename}, {"_id": 0, "file_id": 1})
    if not document:
        raise HTTPException(status_code=404, detail="File not found")
    file_id = document.get("file_id")
    # Delete all chunks with the same file_id
    knowledge_collection.delete_many({"file_id": file_id})
    # Delete the local file
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"message": f"File '{filename}' deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
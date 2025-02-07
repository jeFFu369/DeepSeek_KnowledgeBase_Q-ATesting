import os
import PyPDF2
from docx import Document
import requests
import re
from flask import Flask, request, render_template, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flashing messages

# Global variable to store the knowledge base
knowledge_base = ""

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text    

# Function to extract text from DOCX
def query_local_model(prompt, context):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "deepseek-r1:1.5b",
        "prompt": f"{context}\n\nQuestion: {prompt}\nProvide complete and accurate answer\nAnswer:",
        "stream": False,
        "max_tokens": 500  # Increase the token limit to allow longer responses
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        response_data = response.json()
        raw_response = response_data.get("response", "").strip()

        # Log the raw response for debugging
        print(f"Raw response from model: {raw_response}")

        # Post-process the response
        cleaned_response = clean_response(raw_response)
        print(f"Cleaned response: {cleaned_response}")  # Log cleaned response

        # Validate the response
        if not validate_response(cleaned_response):
            print(f"Debug: Invalid response - {raw_response}")  # Log raw response for debugging
            return "I'm sorry, I couldn't find a clear answer to your question."

        return cleaned_response
    else:
        raise Exception(f"Error querying local model: {response.status_code}, {response.text}")

def clean_response(response):
    """
    Clean up the model's response while preserving Markdown formatting.
    """
    # Remove <think> tags and internal reasoning
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    # Check if the response is truncated
    if response.endswith(("...", "Dete", "calcu")):  # Common truncation patterns
        response += "\n(Note: The response may be incomplete due to length restrictions.)"

    # Preserve Markdown headers (e.g., ### Conclusion)
    response = re.sub(r"^(#+)\s+", r"\1 ", response, flags=re.MULTILINE)  # Normalize header spacing

    # Remove code blocks and bold formatting only if necessary
    response = re.sub(r"```[\s\S]*?```", "", response)  # Remove code blocks
    response = re.sub(r"\*\*(.*?)\*\*", r"\1", response)  # Remove bold formatting

    # Remove excessive newlines but preserve structure
    response = re.sub(r"\n{3,}", "\n\n", response).strip()

    return response

def validate_response(response):
    """
    Validate the response to ensure it meets basic criteria.
    """
    valid_keywords = [
        "steps", "complete", "task", "follow", "summary",
        "how to", "guide", "instructions", "process", "solution",
        "implement", "develop", "build", "create", "setup"
    ]
    for keyword in valid_keywords:
        if keyword in response.lower():
            return True
    return False

# Route to display the home page
@app.route("/", methods=["GET", "POST"])
def index():
    global knowledge_base
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        # Save the uploaded file temporarily
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)  # Create the uploads directory if it doesn't exist
        file.save(file_path)

        # Extract text based on file type
        try:
            if file.filename.endswith(".pdf"):
                knowledge_base = extract_text_from_pdf(file_path)
            elif file.filename.endswith(".docx"):
                knowledge_base = extract_text_from_docx(file_path)
            else:
                flash("Unsupported file format. Please upload a PDF or DOCX file.", "error")
                return redirect(request.url)

            flash("Knowledge base loaded successfully!", "success")
        except Exception as e:
            flash(f"Error processing file: {e}")
            return redirect(request.url)

    return render_template("index.html")

# Route to handle user queries
@app.route("/query", methods=["POST"])
def query():
    global knowledge_base
    if not knowledge_base:
        flash("Please upload a knowledge base first.")
        return redirect(url_for("index"))

    question = request.form.get("question")
    if not question:
        flash("Please provide a question.")
        return redirect(url_for("index"))

    try:
        answer = query_local_model(question, knowledge_base)
        return render_template("result.html", question=question, answer=answer)
    except Exception as e:
        flash(f"An error occurred: {e}")
        return redirect(url_for("index"))

# HTML templates
# Create a folder named "templates" in the same directory as this script
# Inside "templates", create two files: "index.html" and "result.html"

if __name__ == "__main__":
    app.run(debug=True)
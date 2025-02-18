# Knowledge Base Query Application

## Overview

The **Knowledge Base Query Application** is a Flask-based web application designed to allow users to upload PDF or DOCX documents as knowledge bases and query them using a local AI model. The app extracts text from uploaded files, stores it in memory, and uses the AI model to generate accurate and context-aware responses to user questions.

Key features:
- Upload and process PDF/DOCX documents as knowledge bases.
- Extract text from uploaded files and store it for querying.
- Query the knowledge base using a local AI model (`deepseek-r1:1.5b`).
- Auto-clean and validate AI-generated responses for clarity and accuracy.
- User-friendly interface with feedback messages.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Usage](#usage)
   - [Running the Application](#running-the-application)
   - [Uploading Documents](#uploading-documents)
   - [Querying the Knowledge Base](#querying-the-knowledge-base)
6. [API Endpoints](#api-endpoints)
7. [Dependencies](#dependencies)
8. [Contributing](#contributing)
9. [License](#license)

---

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.8 or higher
- A local AI model server running at `http://localhost:11434/api/generate` (e.g., Ollama with a model like `deepseek-r1:1.5b`)
- Flask and other required Python libraries (see [Dependencies](#dependencies))

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/KnowledgeBaseQueryApp.git
   cd KnowledgeBaseQueryApp
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create Required Directories**:
   Ensure the `uploads` directory exists for storing uploaded files:
   ```bash
   mkdir uploads
   ```

---

## Project Structure

```
KnowledgeBaseQueryApp/
├── app.py                  # Main Flask application
├── requirements.txt        # List of Python dependencies
├── uploads/                # Directory for storing uploaded files
├── templates/              # Jinja2 HTML templates for the frontend
│   ├── index.html          # Home page template
│   └── result.html         # Query result template
└── README.md               # This file
```

---

## Configuration

1. **Local AI Model Server**:
   Ensure the local AI model server is running at `http://localhost:11434/api/generate`. Update the URL in the `query_local_model` function if necessary.

2. **Secret Key**:
   The Flask app uses a secret key for flashing messages. You can change it in the `app.secret_key` variable in `app.py`:
   ```python
   app.secret_key = "your-secret-key"
   ```

---

## Usage

### Running the Application

Start the Flask application:
```bash
python app.py
```

The app will be accessible at `http://127.0.0.1:5000`.

### Uploading Documents

1. Navigate to the home page (`http://127.0.0.1:5000`).
2. Use the upload form to select a PDF or DOCX file.
3. Click **Upload** to process and store the file as the knowledge base.

### Querying the Knowledge Base

1. After uploading a document, enter your question in the input field on the home page.
2. Click **Submit** to get an AI-generated response based on the uploaded knowledge base.

---

## API Endpoints

| Endpoint       | Method | Description                                      |
|----------------|--------|--------------------------------------------------|
| `/`            | GET    | Home page with upload form and query interface   |
| `/`            | POST   | Upload a new knowledge base document             |
| `/query`       | POST   | Submit a query and receive an AI-generated answer|

---

## Dependencies

Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

List of dependencies:
- `Flask`
- `PyPDF2`
- `python-docx`
- `requests`
- `re`

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature/fix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Let me know if you'd like to add or modify anything in the README! 😊

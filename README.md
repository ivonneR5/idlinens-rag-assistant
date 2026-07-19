# IDLinens RAG Assistant

## Description

IDLinens RAG Assistant is an intelligent assistant based on Retrieval-Augmented Generation (RAG), developed as the final project for the Oracle Next Education (ONE) + Alura AI Agents program.

The assistant answers questions about the **IDLinens HA** hospital platform using only the official documentation indexed in its knowledge base. This approach minimizes hallucinations and provides references to the documentation used to generate each response.

---

# Project Objective

Develop an Artificial Intelligence assistant capable of understanding natural language questions and generating accurate answers based exclusively on the official technical and user documentation of the IDLinens HA platform.

---

# Main Features

- Semantic document search using embeddings.
- FAISS vector database.
- Response generation using Gemma 3 running locally through Ollama.
- Streamlit web interface.
- Automatic citation of the consulted documentation.
- Support for Spanish documentation.
- Local execution without commercial AI APIs.
- Retrieval-Augmented Generation (RAG) architecture.

---

# System Architecture

```text
PDF Documentation
        │
        ▼
Document Loading
        │
        ▼
Chunking
        │
        ▼
Sentence Transformers Embeddings
        │
        ▼
FAISS Vector Store
        │
        ▼
Semantic Retrieval
        │
        ▼
Gemma 3 (Ollama)
        │
        ▼
Answer Generation
```

---

# Documentation Used

The assistant indexes the following official documentation:

- Operational Manual – IDLinens HA
- Android Application User Manual
- Dashboard User Manual

Knowledge Base Statistics

- 3 PDF documents
- 137 indexed pages
- 288 indexed chunks

---

# Technologies

- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Sentence Transformers
- Ollama
- Gemma 3 (4B)
- Streamlit
- Oracle Cloud Infrastructure (OCI)

---

# Installation

Clone the repository

```bash
git clone https://github.com/ivonneR5/idlinens-rag-assistant.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Build the Vector Database

```bash
python src/chunk_documents.py

python src/create_vector_store.py
```

---

# Run the Application

```bash
streamlit run src/app.py
```

---

# Example Questions

- How do I register a new linen?
- How do I send linens to the laundry?
- How do I change the RFID reader power?
- How do I create a Dashboard user?
- How do I check an asset history?

---

# Project Structure

```text
docs/
src/
evidencias/
requirements.txt
README.md
LICENSE
```

---

# Deployment

The application was deployed on an Oracle Cloud Infrastructure virtual machine.

Deployment environment:

- Ubuntu Server
- Oracle Cloud Infrastructure (OCI)
- Ollama
- Gemma 3
- Streamlit
- Nginx
- FAISS Vector Store

---

# Deployment Evidence

The repository includes deployment screenshots inside the **evidencias/** directory, showing:

- Oracle Cloud virtual machine.
- Running application.
- AI assistant answering questions.
- Source citations.
- Running services.

---
## Demonstration Videos

A complete project presentation and functional demonstration are available at the following link:

[View demonstration videos] 
https://drive.google.com/drive/folders/1e7xvCnO4fuUTyokYmhY-KuR_7blNXolp?usp=drive_link 


# Current Limitations

The response time depends on the available hardware resources.

When the assistant runs on an **Oracle Cloud Always Free** instance, responses may take longer because the Gemma 3 language model runs locally on limited CPU resources.

Despite this limitation, the assistant continues providing accurate responses based exclusively on the indexed documentation.

---

# Future Improvements

- Hybrid search (semantic + keyword search).
- Result re-ranking.
- Incremental vector database updates.
- OCR support for scanned documents.
- Multi-hospital support.
- Conversation memory.

---

# Project Status

✅ Completed

---

# Author

**Ivonne Negrete**

Final project developed for the **Oracle Next Education (ONE)** + **Alura AI Agents** program under the challenge **"Build an Intelligent Agent using RAG"**.

Licensed under the MIT License.
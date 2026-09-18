# Intelligent Research Paper Summarizer and Q&A System

**Research Paper AI** is a research-paper analysis application that lets users upload papers, generate summaries, and interact with the content through conversational question answering.

The system combines document processing, information retrieval, and Large Language Models (LLMs) to produce context-grounded responses based on the uploaded paper.

---

## Live Application

[Research Paper AI - Live Demo](https://research-paper-ai-online.onrender.com/)

Upload a research paper and interact with it through the conversational interface.

---

## Overview

Research papers are long, technical, and slow to read. Research Paper AI provides an interactive way to explore a paper without manually searching every page.

The application allows users to:

- Upload a research paper in PDF format
- Extract and process the paper's text
- Divide the document into manageable chunks
- Retrieve relevant sections based on user questions
- Generate AI responses using the retrieved context
- Ask multiple questions conversationally
- Generate and understand research-paper summaries

---

## Key Features

- **Research Paper Upload** - Upload papers in PDF format
- **Automatic Text Extraction** - Extract text from uploaded PDFs
- **Intelligent Text Chunking** - Split documents into smaller meaningful sections
- **Relevant Context Retrieval** - Identify relevant content for each question
- **AI-Powered Answers** - Generate responses using an LLM
- **Conversational Q&A** - Ask multiple questions about the uploaded paper
- **Research Paper Summarization** - Understand lengthy papers faster
- **Context-Grounded Responses** - Answers are based on retrieved paper content
- **Modern Dark-Themed Interface** - Clean conversational UI
- **Single-Service Deployment** - Frontend and backend deployed together
- **Secure API Configuration** - API keys managed through environment variables

---

## System Architecture

```text
                        User
            (Upload PDF / Ask Questions)
                         |
                         v
                  React Frontend
         (Chat Interface, Upload, Conversations)
                         |
                         v
                  FastAPI Backend
            (API Endpoints, Request Handling)
                         |
            +------------+------------+
            |                         |
            v                         v
     PDF Ingestion              Retrieval
   (Text Extraction,        (Relevant Chunks,
    Cleaning, Chunking)      Context Selection)
            |                         |
            +------------+------------+
                         |
                         v
                        LLM
           (Context-Grounded Generation)
                         |
                         v
                     Response
                (Answer + Context)
```

---

## Processing Pipeline

```text
Research Paper PDF
        |
        v
  Text Extraction
        |
        v
   Text Cleaning
        |
        v
   Text Chunking
        |
        v
Relevant Context Retrieval
        |
        v
LLM Prompt Construction
        |
        v
Context-Grounded Generation
        |
        v
    AI Response
```

---

## Technology Stack

**Frontend**
- React
- TypeScript
- Vite
- Tailwind CSS

**Backend**
- Python
- FastAPI
- Uvicorn

**AI / NLP**
- Large Language Models (LLMs)
- Retrieval-Augmented Generation concepts
- Keyword-based document retrieval
- Natural Language Processing
- Context-grounded generation

**Document Processing**
- PyMuPDF
- PDF text extraction
- Text cleaning
- Text chunking

**Deployment**
- Docker
- Render

---

## Project Structure

```text
research-paper-ai-online/
|
├── backend/
│   └── main.py
|
├── ingestion/
│   └── ingest.py
|
├── retrieval/
│   └── rag.py
|
├── data/
│   ├── papers/
│   └── processed/
|
├── src/
│   ├── components/
│   ├── pages/
│   └── ...
|
├── public/
|
├── Dockerfile
├── requirements.txt
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── README.md
└── README_SETUP.md
```

---

## How It Works

**1. Upload a Research Paper**  
The user uploads a paper in PDF format through the web interface.

**2. Extract Text**  
The backend processes the PDF and extracts selectable text using PyMuPDF.

**3. Clean and Process the Document**  
Extracted text is cleaned and prepared for further processing.

**4. Create Chunks**  
The paper is divided into smaller text chunks, so the system works with specific portions instead of the full document for every question.

**5. Retrieve Relevant Information**  
When a question is asked, the retrieval component identifies relevant portions of the paper.

**6. Construct the LLM Context**  
Retrieved content is passed as context to the language model.

**7. Generate an AI Response**  
The model generates a response grounded in the retrieved content.

**8. Continue the Conversation**  
Follow-up questions can be asked through the conversational interface.

---

## Example Questions

- What is the main contribution of this paper?
- What problem does the paper attempt to solve?
- Explain the methodology used by the authors.
- What dataset was used?
- What are the main experimental results?
- What are the limitations of this approach?
- Explain this paper in simple terms.
- How does the proposed method compare with previous approaches?
- What is the conclusion of the paper?

---

## Example Workflow

```text
User uploads: Proximal Policy Optimization Algorithms.pdf
        |
        v
System processes the paper
  PDF -> Text Extraction -> Text Cleaning -> Text Chunking
        |
        v
User asks: What is the main idea behind PPO?
        |
        v
Retrieval system identifies relevant paper content
        |
        v
Retrieved content is provided to the LLM
        |
        v
User receives an AI-generated response (Answer + Relevant Context)
```

---

## Retrieval-Augmented Generation Approach

The project follows a retrieval-based generation architecture. Instead of asking the language model to answer without document context, the system first identifies relevant information from the uploaded paper.

```text
Research Paper
      |
      v
Text Extraction
      |
      v
   Chunking
      |
      v
   Retrieval
      |
      v
Relevant Paper Context
      |
      v
  LLM Prompt
      |
      v
Context-Grounded Answer
```

---

## Project Objectives

- Reduce the time required to understand lengthy research papers
- Provide an interactive way to explore research documents
- Enable conversational question answering over research papers
- Retrieve relevant information before generating responses
- Ground AI responses in content from the uploaded document
- Make complex research content easier to understand
- Demonstrate a practical application of Generative AI and information retrieval

---

## AI Pipeline

```text
Document
   |
   v
PDF Extraction
   |
   v
Text Cleaning
   |
   v
Chunking
   |
   v
Information Retrieval
   |
   v
Relevant Context
   |
   v
Prompt Construction
   |
   v
Large Language Model
   |
   v
Generated Response
```

---

## Environment Variables

API keys and other sensitive values are stored as environment variables.

```env
GROQ_API_KEY=your_api_key
OPENROUTER_API_KEY=your_api_key
```

**Security**

Never commit real API keys, passwords, or tokens to GitHub. The local `.env` file must remain excluded from version control. An `.env.example` file is provided as a reference for required variables.

---

## Running with Docker

Build the image:

```bash
docker build -t research-paper-ai .
```

Run the application:

```bash
docker run -p 10000:10000 research-paper-ai
```

The application is then accessible locally on the configured port.

---

## Local Development

**Backend**

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn backend.main:app --reload
```

**Frontend**

Install packages:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

---

## Deployment

The project runs as a single web service. The Docker deployment contains:

```text
React Frontend
      +
FastAPI Backend
      +
PDF Processing
      +
Retrieval System
      +
LLM Integration
```

Frontend and backend are deployed together as one application, configured for Render.

---

## Future Improvements

- Multi-paper comparison
- Automatic paper metadata extraction
- Citation extraction
- Improved semantic retrieval
- Conversation memory
- Highlighting relevant passages
- Section-aware retrieval
- Table and figure understanding
- Multimodal PDF processing
- Improved summarization strategies
- User authentication
- Persistent chat history
- Research-paper recommendation
- Export conversations as PDF

---

## Academic Project

This project was developed as an Artificial Intelligence and Machine Learning academic project, applying Generative AI, information retrieval, natural language processing, and document-processing techniques to research-paper analysis.

It demonstrates how Large Language Models can be combined with document retrieval to create an interactive research assistant.

---

## Author

**Pothuri Indraneel**  
B.Tech - Artificial Intelligence and Machine Learning  
Woxsen University, Hyderabad

---

## Project Highlights

- Research paper processing
- PDF text extraction
- Intelligent text chunking
- Information retrieval
- LLM-based generation
- Conversational Q&A
- Research paper summarization
- Context-grounded responses
- Docker deployment
- FastAPI backend
- React frontend
- Online web application

---

## License

This project is intended for academic and educational purposes.

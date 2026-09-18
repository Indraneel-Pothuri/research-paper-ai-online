"""Production FastAPI backend for Research Paper AI.

The same FastAPI process serves:
- the React single-page application
- the RAG API
- paper upload/indexing
- paper comparison
- paper summarization

This keeps the project deployable as one Docker web service with one public URL.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ingestion import ingest
from retrieval import rag
from retrieval.llm_provider import generate_with_fallback

BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "dist"
PAPERS_DIR = BASE_DIR / "data" / "papers"
EMBEDDINGS_FILE = BASE_DIR / os.getenv(
    "EMBEDDINGS_FILE", "data/processed/research_embeddings.json"
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

app = FastAPI(
    title="Research Paper AI",
    description="Grounded RAG API for research-paper question answering.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Schemas
# ============================================================

class ChatMessageSchema(BaseModel):
    role: str
    content: str


class ChatRequestSchema(BaseModel):
    conversation_id: Optional[str] = None
    message: str = Field(min_length=1)
    paper_scope_type: Optional[str] = "all"
    paper_scope_id: Optional[str] = None
    history: List[ChatMessageSchema] = Field(default_factory=list)


class CompareRequestSchema(BaseModel):
    paper_id_a: str
    paper_id_b: str


class SummarizeRequestSchema(BaseModel):
    paper_id: str


# ============================================================
# Helpers
# ============================================================

def extract_json(text: str) -> Any:
    """Extract a JSON object/array from normal or fenced LLM output."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            return value
        except json.JSONDecodeError:
            continue

    raise ValueError("LLM did not return valid JSON.")


def safe_filename(filename: str) -> str:
    """Return a filesystem-safe PDF filename."""
    name = Path(filename or "research-paper.pdf").name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name or "research-paper.pdf"


def persist_embeddings() -> None:
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = EMBEDDINGS_FILE.with_suffix(".tmp")
    with temp_file.open("w", encoding="utf-8") as handle:
        json.dump(rag.documents, handle, ensure_ascii=False)
    temp_file.replace(EMBEDDINGS_FILE)


def get_paper_metadata_list() -> List[Dict[str, Any]]:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    papers: List[Dict[str, Any]] = []

    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        paper_id = ingest.create_paper_id(pdf_path)
        file_size_mb = f"{pdf_path.stat().st_size / (1024 * 1024):.1f} MB"
        mod_time = time.strftime("%b %d, %Y", time.localtime(pdf_path.stat().st_mtime))

        title = pdf_path.stem.replace("_", " ")
        abstract = ""
        page_count = 0

        try:
            with fitz.open(pdf_path) as document:
                page_count = len(document)
                first_page_text = document[0].get_text("text") if page_count else ""

            lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
            if lines:
                candidate = lines[0]
                if 5 <= len(candidate) <= 180:
                    title = candidate

            abstract_match = re.search(
                r"abstract\s*[:\-–—]?\s*(.*?)(?=\n\s*(?:1\.?\s+)?introduction\b|\Z)",
                first_page_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if abstract_match:
                abstract = re.sub(r"\s+", " ", abstract_match.group(1)).strip()[:500]
                if abstract and len(abstract) >= 40:
                    abstract += "..."
        except Exception as exc:
            print(f"[Metadata] {pdf_path.name}: {exc}")

        papers.append(
            {
                "id": paper_id,
                "title": title,
                "filename": pdf_path.name,
                "uploadDate": mod_time,
                "pageCount": page_count,
                "status": "Ready",
                "fileSize": file_size_mb,
                "collectionIds": [],
                "abstract": abstract or f"Research paper document '{pdf_path.name}'.",
            }
        )

    return papers


def find_paper(paper_id: str) -> Optional[Dict[str, Any]]:
    return next((paper for paper in get_paper_metadata_list() if paper["id"] == paper_id), None)


def paper_chunks(paper_id: str) -> List[Dict[str, Any]]:
    return [doc for doc in rag.documents if doc.get("paper_id") == paper_id]


def select_paper_context(paper_id: str, max_chunks: int = 8) -> List[Dict[str, Any]]:
    """Select a useful spread of chunks from a single paper."""
    chunks = paper_chunks(paper_id)
    if len(chunks) <= max_chunks:
        return chunks

    # Keep early chunks (title/abstract/introduction) plus evenly spaced chunks.
    indices = {0, 1, 2}
    step = max(1, len(chunks) // max_chunks)
    indices.update(range(0, len(chunks), step))
    selected = [chunks[i] for i in sorted(indices) if i < len(chunks)]
    return selected[:max_chunks]


# ============================================================
# Health / papers
# ============================================================

@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "rag": True,
        "llm": bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY")),
        "chunks_loaded": len(rag.documents),
        "embedding_model": rag.EMBEDDING_MODEL,
    }


@app.get("/api/papers")
def list_papers() -> List[Dict[str, Any]]:
    return get_paper_metadata_list()


@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: str) -> Dict[str, Any]:
    paper = find_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@app.post("/api/papers/upload")
async def upload_paper(file: UploadFile = File(...)) -> Dict[str, Any]:
    filename = safe_filename(file.filename or "research-paper.pdf")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = PAPERS_DIR / filename
    saved_path.write_bytes(content)

    paper_id = ingest.create_paper_id(saved_path)

    try:
        chunks = ingest.process_pdf(saved_path, paper_id)
        if not chunks:
            raise ValueError("No selectable text could be extracted from the PDF.")

        model = rag.get_embedding_model()
        embeddings = model.encode(
            [chunk["text"] for chunk in chunks],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Replace an existing paper with the same ID rather than duplicating it.
        rag.documents[:] = [
            doc for doc in rag.documents if doc.get("paper_id") != paper_id
        ]
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
            rag.documents.append(chunk)

        persist_embeddings()
    except Exception as exc:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {exc}") from exc

    paper = find_paper(paper_id)
    if paper:
        return paper

    return {
        "id": paper_id,
        "title": saved_path.stem.replace("_", " "),
        "filename": saved_path.name,
        "uploadDate": "Just now",
        "pageCount": 0,
        "status": "Ready",
        "fileSize": f"{saved_path.stat().st_size / (1024 * 1024):.1f} MB",
        "collectionIds": [],
        "abstract": "Uploaded and indexed successfully.",
    }


@app.delete("/api/papers/{paper_id}")
def delete_paper(paper_id: str) -> Dict[str, str]:
    paper = find_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    pdf_path = PAPERS_DIR / paper["filename"]
    pdf_path.unlink(missing_ok=True)

    rag.documents[:] = [
        doc for doc in rag.documents if doc.get("paper_id") != paper_id
    ]
    persist_embeddings()

    return {"status": "deleted", "paper_id": paper_id}


# ============================================================
# Chat
# ============================================================

@app.post("/api/chat")
def chat_endpoint(req: ChatRequestSchema) -> Dict[str, Any]:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    history = [{"role": item.role, "content": item.content} for item in req.history]
    retrieval_query = rag.build_retrieval_query(message, history)

    # Retrieve a larger pool first so paper scoping does not accidentally
    # discard the requested paper just because it was not in the first six hits.
    retrieved = rag.retrieve_chunks(retrieval_query, top_k=rag.RETRIEVAL_K)

    if req.paper_scope_type == "paper" and req.paper_scope_id:
        scoped = [
            item for item in retrieved
            if item.get("paper_id") == req.paper_scope_id
        ]

        # If semantic retrieval missed the selected paper, retrieve directly
        # from that paper's chunks using the same embedding model.
        if not scoped:
            scoped = rag.retrieve_chunks_for_paper(
                retrieval_query,
                req.paper_scope_id,
                top_k=rag.TOP_K,
            )
        retrieved = scoped
    else:
        retrieved = retrieved[:rag.TOP_K]

    if not retrieved:
        return {
            "answer": "I couldn't find enough information in your research papers to answer this confidently.",
            "sources": [],
            "provider": "System",
            "model": "RAG Filter",
        }

    if retrieved[0].get("similarity", 0.0) < rag.MIN_SIMILARITY:
        context = (
            "IMPORTANT: Retrieval confidence is low. The evidence below may not directly answer the question.\n\n"
            + rag.build_context(retrieved)
        )
    else:
        context = rag.build_context(retrieved)

    messages = rag.build_messages(message, context, history)

    try:
        llm_result = generate_with_fallback(messages)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="All configured language model providers are currently unavailable.",
        ) from exc

    timestamp = int(time.time() * 1000)
    sources = []
    for index, item in enumerate(retrieved, start=1):
        title = Path(item.get("paper_name", "Research Paper")).stem.replace("_", " ")
        sources.append(
            {
                "id": f"src-{timestamp}-{index}",
                "paperId": item.get("paper_id", "unknown"),
                "paperTitle": title,
                "filename": item.get("paper_name", "paper.pdf"),
                "page": item.get("page", 1),
                "section": item.get("section") or "Section Excerpt",
                "excerpt": item.get("text", "")[:280].strip() + "...",
                "similarity": round(float(item.get("similarity", 0.0)), 4),
                "score": round(float(item.get("score", 0.0)), 4),
            }
        )

    return {
        "answer": llm_result.text,
        "sources": sources,
        "provider": llm_result.provider,
        "model": llm_result.model,
    }


# ============================================================
# Compare / summarize
# ============================================================

@app.post("/api/compare")
def compare_papers_endpoint(req: CompareRequestSchema) -> Dict[str, Any]:
    paper_a = find_paper(req.paper_id_a)
    paper_b = find_paper(req.paper_id_b)
    if not paper_a or not paper_b:
        raise HTTPException(status_code=404, detail="One or both papers were not found.")

    chunks_a = select_paper_context(req.paper_id_a, max_chunks=8)
    chunks_b = select_paper_context(req.paper_id_b, max_chunks=8)

    prompt = f"""
Compare these two research papers using ONLY the supplied excerpts.
If a category is not supported by the excerpts, explicitly say "Not established in the supplied excerpts".
Do not invent datasets, results, limitations, authors, or numbers.

PAPER A: {paper_a['title']} ({paper_a['filename']})

{rag.build_context(chunks_a)}

PAPER B: {paper_b['title']} ({paper_b['filename']})

{rag.build_context(chunks_b)}

Return ONLY valid JSON with this exact shape:
[
  {{"category":"Research Objective","paperAValue":"...","paperBValue":"..."}},
  {{"category":"Core Methodology","paperAValue":"...","paperBValue":"..."}},
  {{"category":"Dataset / Benchmarks","paperAValue":"...","paperBValue":"..."}},
  {{"category":"Key Findings","paperAValue":"...","paperBValue":"..."}},
  {{"category":"Primary Limitations","paperAValue":"...","paperBValue":"..."}}
]
""".strip()

    try:
        result = generate_with_fallback(
            [
                {"role": "system", "content": "You are a research-paper comparison assistant. Return strict JSON only."},
                {"role": "user", "content": prompt},
            ]
        )
        dimensions = extract_json(result.text)
        if not isinstance(dimensions, list):
            raise ValueError("Comparison response was not a JSON array.")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The comparison model did not return a valid result.") from exc

    return {"paperA": paper_a, "paperB": paper_b, "dimensions": dimensions}


@app.post("/api/summarize")
def summarize_paper_endpoint(req: SummarizeRequestSchema) -> Dict[str, Any]:
    paper = find_paper(req.paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    chunks = select_paper_context(req.paper_id, max_chunks=10)
    context = rag.build_context(chunks)

    prompt = f"""
Create a grounded summary of the research paper below.
Use ONLY the supplied evidence. If something is not supported, say "Not established in the supplied evidence".
Do not invent authors, dates, datasets, metrics, results, limitations, or future work.

PAPER: {paper['title']} ({paper['filename']})

EVIDENCE:
{context}

Return ONLY valid JSON with:
{{
  "abstract": "...",
  "keyProblem": "...",
  "methodology": "...",
  "datasetExperiments": "...",
  "keyFindings": ["...", "...", "..."],
  "limitations": "...",
  "futureDirections": "..."
}}
""".strip()

    try:
        result = generate_with_fallback(
            [
                {"role": "system", "content": "You are a research-paper summarization assistant. Return strict JSON only."},
                {"role": "user", "content": prompt},
            ]
        )
        data = extract_json(result.text)
        if not isinstance(data, dict):
            raise ValueError("Summary response was not a JSON object.")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The summarization model did not return a valid result.") from exc

    return {
        "paperId": paper["id"],
        "paperTitle": paper["title"],
        "authors": paper.get("authors", ["Research Authors"]),
        "year": paper.get("year", 2026),
        "abstract": data.get("abstract", ""),
        "keyProblem": data.get("keyProblem", ""),
        "methodology": data.get("methodology", ""),
        "datasetExperiments": data.get("datasetExperiments", ""),
        "keyFindings": data.get("keyFindings", []),
        "limitations": data.get("limitations", ""),
        "futureDirections": data.get("futureDirections", ""),
    }


# ============================================================
# Production frontend
# ============================================================

@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    index_file = DIST_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=503, detail="Frontend build is not available.")
    return FileResponse(index_file)


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str) -> FileResponse:
    """Serve the React SPA for client-side routes."""
    requested = DIST_DIR / full_path
    if requested.is_file():
        return FileResponse(requested)

    index_file = DIST_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_file)

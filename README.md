# Research Paper AI

An intelligent research-paper summarizer and conversational Q&A application using semantic retrieval and grounded LLM generation.

## Deploy

See **README_DEPLOY.md** for the one-link Docker deployment setup.

The application is designed to run as one web service with the React frontend and FastAPI backend together.

## Main pipeline

PDF -> text extraction -> overlapping chunks -> Sentence-Transformer embeddings -> semantic retrieval -> grounded LLM answer -> paper/page sources

## LLM fallback

1. Groq
2. OpenRouter
3. LM Studio (local development fallback)

API keys belong in environment variables, never in source control.

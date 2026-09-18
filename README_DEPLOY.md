# Research Paper AI — deploy as one public web app

This version is packaged as a **single Docker web service**:

- React + Vite frontend
- FastAPI backend
- local Sentence-Transformer embeddings
- grounded RAG retrieval
- Groq -> OpenRouter -> LM Studio fallback for local development
- PDF upload, indexing, chat, comparison and summarization

The frontend calls `/api/...` on the same host, so the deployed application needs only **one public URL**.

## Deploy on Render

1. Create a GitHub repository and push this project.
2. On Render, create a **New → Web Service** and connect the GitHub repository.
3. Choose **Docker** as the runtime.
4. Render will use the included `Dockerfile`.
5. Add these environment variables in Render:

```text
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
```

The model defaults are already in `render.yaml` / the application:

```text
GROQ_MODEL=openai/gpt-oss-20b
OPENROUTER_MODEL=openrouter/free
```

6. Deploy.
7. Render will provide a public URL similar to:

```text
https://research-paper-ai-xxxx.onrender.com
```

Open that URL. The React UI and API are served by the same application.

### Important

Do **not** upload `.env` or API keys to GitHub. Use Render Environment Variables.

The project already contains research PDFs and the generated embedding index. They are included in the Docker image, so the initial papers are available after deployment.

PDF uploads are stored in the container filesystem. On infrastructure where the filesystem is ephemeral, uploaded files can disappear after a service restart/redeploy. If persistent user uploads are required, attach persistent storage or move the paper/index store to durable storage.

## Local development

Create `.env` from `.env.example`, then run:

```bash
npm install
npm run dev
```

For the backend:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

For split local development, set:

```text
VITE_API_URL=http://127.0.0.1:8000
```

For the production Docker build, leave `VITE_API_URL` empty so the browser uses the same origin.

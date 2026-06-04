# Deployment

Two pieces deploy separately:

- **Backend** (FastAPI + embedded Kùzu + sqlite-vec + local bge-small) → a container host (Render / Railway / Fly). It **cannot** run on Vercel serverless.
- **Frontend** (Next.js) → Vercel.

## 1. Push to GitHub

```bash
git add .
git commit -m "Add deploy scaffold"
git push
```

## 2. Deploy the backend container

The repo ships a root `Dockerfile` (`python:3.11-slim`, runs `uvicorn backend.app:app`).

### Prebuilt-data caveat (IMPORTANT)

`data/` (the Kùzu graph DB and the sqlite-vec store) is **.gitignored** and built
offline, so it is **not** in the GitHub repo. The Docker image `COPY data/ ./data/`
therefore needs the data present at build time. Either:

- Build `data/` locally (run `scripts/ingest_corpus.py`) **before** building/deploying so the
  files exist in the build context, **or**
- Uncomment the ingest step in the `Dockerfile` to build `data/` during the image build.

Without this the container starts but has an empty/missing graph + vector store.

### Render (uses `render.yaml`)

1. New → Blueprint → point at the repo. Render reads `render.yaml` (Docker web service).
2. Set the `GROQ_API_KEY` env var in the dashboard (it's declared `sync: false`).
3. Deploy. Render injects `PORT`; the container's `CMD` already honors `${PORT:-8000}`.

### Railway / Fly (alternative)

- Both auto-detect the root `Dockerfile`. Set `GROQ_API_KEY` and bind to `$PORT`
  (the `CMD` already does). No code changes needed.

Note the backend's public URL once live (e.g. `https://graphrag-backend.onrender.com`).

## 3. Deploy the frontend on Vercel

1. New Project → import the repo → set **Root Directory** to `frontend/`.
   Vercel auto-detects Next.js (`frontend/vercel.json` is present).
2. Set the env var **`NEXT_PUBLIC_API_URL`** to the deployed backend URL from step 2
   (e.g. `https://graphrag-backend.onrender.com`). It defaults to
   `http://localhost:8000`, which only works for local dev, so this **must** be set
   for production.
3. Deploy.

## 4. CORS

The backend already allows all origins (`*`), so no extra CORS config is needed for
the Vercel frontend.

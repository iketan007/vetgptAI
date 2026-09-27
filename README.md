# VetGPT — AI-Powered Pet Care Assistant

A RAG-based veterinary chatbot: LangChain + FAISS retrieval over a curated
pet-care knowledge base, generation via Llama 2, served through a FastAPI
backend, with a simple chat frontend.

```
vetgpt-project/
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI app (/chat, /health, /reindex)
│   │   ├── rag.py         # embeddings, FAISS index, prompt, LLM call
│   │   └── config.py      # all settings, via environment variables
│   ├── data/
│   │   ├── knowledge_base/    # curated .md vet-care docs (edit/add freely)
│   │   └── faiss_index/       # generated on first run (git-ignored)
│   ├── scripts/build_index.py # rebuild the FAISS index manually
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
└── frontend/
    └── index.html          # chat UI, calls the backend's /chat endpoint
```

## 1. Run the backend locally

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

You need a Llama 2 model to generate answers. Pick ONE of these in `.env`:

- **Local GGUF model (default, `LLM_PROVIDER=llama_cpp`)** — download a
  quantized Llama 2 chat model (e.g. `llama-2-7b-chat.Q4_K_M.gguf` from
  TheBloke on Hugging Face), place it at the path in `LLAMA_MODEL_PATH`
  (default `./models/llama-2-7b-chat.Q4_K_M.gguf`).
- **Ollama** — `ollama pull llama2`, then set `LLM_PROVIDER=openai_compatible`
  and leave `OPENAI_COMPATIBLE_BASE_URL=http://localhost:11434/v1`.
- **Hosted HF Inference Endpoint** — set `LLM_PROVIDER=hf_endpoint` and fill
  in `HF_ENDPOINT_URL` / `HF_API_TOKEN`.

Then start the API:

```bash
uvicorn app.main:app --reload --port 8000
```

The first request builds the FAISS index automatically from
`data/knowledge_base/*.md` (takes a few seconds). To rebuild it manually
after editing the knowledge base:

```bash
python scripts/build_index.py
```

Test it:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How often should I feed my puppy?"}'
```

## 2. Run the frontend

`frontend/index.html` is a single self-contained file — just open it in a
browser, or serve it:

```bash
cd frontend
python -m http.server 5500
```

It calls `http://localhost:8000` by default. To point it at a different
backend URL (e.g. once deployed), add this before the closing `</body>` tag:

```html
<script>window.VETGPT_API_BASE_URL = 'https://your-deployed-backend.com';</script>
```

## 3. Deploying for a public link

The backend needs a host that can run a Python process (and ideally a GPU
or enough RAM for the quantized model):

- **Render / Railway / Fly.io** — deploy the `backend/Dockerfile`, mount or
  bake in your GGUF model file, set env vars from `.env.example`.
- **Hugging Face Spaces (Docker SDK)** — good free option for demos.
- **Any VM (EC2, DigitalOcean, etc.)** — clone the repo, follow step 1.

Once the backend has a public URL, deploy `frontend/index.html` anywhere
static (GitHub Pages, Netlify, Vercel, S3) with `VETGPT_API_BASE_URL` set
to that URL — then that static link is your public, shareable chatbot.

## Extending the knowledge base

Drop more `.md` (or adjust the loader glob for `.txt`/`.pdf`) files into
`backend/data/knowledge_base/`, then call `POST /reindex` or re-run
`scripts/build_index.py`.

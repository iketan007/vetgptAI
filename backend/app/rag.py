"""
RAG (Retrieval-Augmented Generation) pipeline for VetGPT.

Flow:
  1. Curated veterinary knowledge base (data/knowledge_base/*.md) is chunked
     and embedded with a Multilingual E5 embedding model.
  2. Chunks are indexed in FAISS for fast semantic similarity search.
  3. At query time, the top-k most relevant chunks are retrieved and stitched
     into a grounded prompt.
  4. The prompt is sent to a Llama 2 model (local GGUF via llama.cpp by
     default; swappable for a hosted endpoint) to produce the final answer.
"""
import os
from typing import List, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document

from app import config

_embeddings = None
_vectorstore = None
_llm = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def build_index() -> FAISS:
    """
    Loads every document in the knowledge base, splits it into chunks,
    embeds them, and builds/saves a FAISS index to disk.
    Run this via scripts/build_index.py whenever the knowledge base changes.
    """
    loader = DirectoryLoader(
        config.KNOWLEDGE_BASE_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs: List[Document] = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)

    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    store.save_local(config.FAISS_INDEX_DIR)
    print(f"Indexed {len(chunks)} chunks from {len(raw_docs)} documents "
          f"-> {config.FAISS_INDEX_DIR}")
    return store


def get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        if not os.path.isdir(config.FAISS_INDEX_DIR):
            _vectorstore = build_index()
        else:
            _vectorstore = FAISS.load_local(
                config.FAISS_INDEX_DIR,
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
    return _vectorstore


def retrieve(query: str, k: int = None) -> List[Document]:
    k = k or config.RETRIEVER_TOP_K
    store = get_vectorstore()
    return store.similarity_search(query, k=k)


SYSTEM_PROMPT = """You are VetGPT, an AI-powered pet care assistant. \
Answer the user's question using ONLY the context provided below, plus general, \
safe pet-care knowledge to fill small gaps. Be warm, clear, and concise. \
If the question concerns a serious symptom, medication, dosage, or emergency, \
clearly recommend contacting a licensed veterinarian immediately rather than \
giving a definitive diagnosis.

Context:
{context}
"""


def build_prompt(query: str, docs: List[Document]) -> str:
    context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "No matching context found."
    system = SYSTEM_PROMPT.format(context=context)
    return f"{system}\n\nUser question: {query}\n\nVetGPT answer:"


def get_llm():
    """
    Returns a LangChain-compatible LLM based on LLM_PROVIDER.
    Swap providers purely via environment variables — no code changes needed.
    """
    global _llm
    if _llm is not None:
        return _llm

    provider = config.LLM_PROVIDER

    if provider == "llama_cpp":
        from langchain_community.llms import LlamaCpp
        _llm = LlamaCpp(
            model_path=config.LLAMA_MODEL_PATH,
            n_ctx=config.LLAMA_N_CTX,
            n_gpu_layers=config.LLAMA_N_GPU_LAYERS,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_NEW_TOKENS,
            verbose=False,
        )

    elif provider == "hf_endpoint":
        from langchain_community.llms import HuggingFaceEndpoint
        _llm = HuggingFaceEndpoint(
            endpoint_url=config.HF_ENDPOINT_URL,
            huggingfacehub_api_token=config.HF_API_TOKEN,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
        )

    elif provider == "openai_compatible":
        # Works with any OpenAI-compatible server: vLLM, Ollama (llama2 model), LM Studio, etc.
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            base_url=config.OPENAI_COMPATIBLE_BASE_URL,
            api_key=config.OPENAI_COMPATIBLE_API_KEY,
            model=config.OPENAI_COMPATIBLE_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_NEW_TOKENS,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

    return _llm


def answer_query(query: str) -> Tuple[str, List[Document]]:
    """
    Full RAG call: retrieve relevant chunks, build a grounded prompt,
    generate the answer, and return it along with the sources used
    (so the frontend can show citations if desired).
    """
    docs = retrieve(query)
    prompt = build_prompt(query, docs)
    llm = get_llm()

    result = llm.invoke(prompt)
    text = result.content if hasattr(result, "content") else str(result)
    return text.strip(), docs

"""
Lightweight RAG pipeline for VetGPT.

Flow:
1. Load veterinary knowledge-base Markdown files.
2. Split documents into chunks.
3. Retrieve relevant chunks using BM25.
4. Send the grounded prompt to Groq through the
   OpenAI-compatible API.
"""

from typing import List, Tuple

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI

from rank_bm25 import BM25Okapi

from app import config


_documents = None
_bm25 = None
_llm = None


def load_documents() -> List[Document]:
    """Load and split the veterinary knowledge base."""

    global _documents

    if _documents is not None:
        return _documents

    loader = DirectoryLoader(
        config.KNOWLEDGE_BASE_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    _documents = splitter.split_documents(raw_docs)

    print(
        f"Loaded {len(raw_docs)} documents "
        f"and created {len(_documents)} chunks."
    )

    return _documents


def get_bm25():
    """Create the BM25 retriever."""

    global _bm25

    if _bm25 is None:
        documents = load_documents()

        tokenized_documents = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        _bm25 = BM25Okapi(tokenized_documents)

    return _bm25


def retrieve(query: str, k: int = None) -> List[Document]:
    """Retrieve the most relevant knowledge-base chunks."""

    k = k or config.RETRIEVER_TOP_K

    documents = load_documents()
    bm25 = get_bm25()

    tokenized_query = query.lower().split()

    results = bm25.get_top_n(
        tokenized_query,
        documents,
        n=k,
    )

    return results


SYSTEM_PROMPT = """You are VetGPT, an AI-powered pet care assistant.

Answer the user's question using the veterinary context provided below.

Be warm, clear, concise, and practical.

If the question concerns a serious symptom, medication, dosage,
poisoning, injury, or emergency, clearly recommend contacting
a licensed veterinarian immediately rather than giving a definitive
diagnosis or unsafe treatment instructions.

Context:
{context}
"""


def build_prompt(query: str, docs: List[Document]) -> str:

    context = (
        "\n\n---\n\n".join(doc.page_content for doc in docs)
        if docs
        else "No matching context found."
    )

    system = SYSTEM_PROMPT.format(context=context)

    return (
        f"{system}\n\n"
        f"User question: {query}\n\n"
        f"VetGPT answer:"
    )


def get_llm():
    """Create the Groq LLM using the OpenAI-compatible API."""

    global _llm

    if _llm is not None:
        return _llm

    if config.LLM_PROVIDER != "openai_compatible":
        raise ValueError(
            "For Render deployment, LLM_PROVIDER must be "
            "'openai_compatible'."
        )

    _llm = ChatOpenAI(
        base_url=config.OPENAI_COMPATIBLE_BASE_URL,
        api_key=config.OPENAI_COMPATIBLE_API_KEY,
        model=config.OPENAI_COMPATIBLE_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_NEW_TOKENS,
    )

    return _llm


def answer_query(query: str) -> Tuple[str, List[Document]]:
    """
    Full RAG pipeline:
    retrieve -> build prompt -> Groq -> answer.
    """

    docs = retrieve(query)

    prompt = build_prompt(query, docs)

    llm = get_llm()

    result = llm.invoke(prompt)

    text = (
        result.content
        if hasattr(result, "content")
        else str(result)
    )

    return text.strip(), docs

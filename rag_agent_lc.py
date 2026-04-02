"""
LangChain + Google Gemini RAG demo: same Chroma + FastEmbed pipeline as rag_agent.py,
generation via ChatGoogleGenerativeAI (langchain-google-genai).
"""

from __future__ import annotations

import lc_transformers_shim  # noqa: F401 — before langchain (optional tokenizer vs. broken torch)

from dotenv import load_dotenv

load_dotenv()

import os
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings
from fastembed import TextEmbedding
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "kb-warranty",
        "title": "Widget Pro — Warranty",
        "text": (
            "Widget Pro carries a 24-month limited warranty from the purchase date. "
            "The warranty covers manufacturing defects. Battery capacity loss under 20% "
            "in year one is considered normal and is not covered. For RMA, email "
            "support@example.com with your serial number."
        ),
    },
    {
        "id": "kb-battery",
        "title": "Widget Pro — Battery care",
        "text": (
            "Charge between 20% and 80% when possible to extend battery lifespan. "
            "Avoid leaving the device at 0% for long periods. Official chargers "
            "only; third-party fast chargers may void the warranty if damage occurs."
        ),
    },
    {
        "id": "kb-api",
        "title": "Acme API — Rate limits",
        "text": (
            "The Acme public API allows 60 requests per minute per API key. "
            "Burst traffic above that returns HTTP 429. Enterprise keys can request "
            "higher limits through sales@acme.example."
        ),
    },
    {
        "id": "kb-onboarding",
        "title": "Acme API — First steps",
        "text": (
            "Create a project in the developer console, then generate an API key. "
            "All requests must send the header X-Acme-Key. Base URL is "
            "https://api.acme.example/v1/."
        ),
    },
]


def _embed_normalize(model: TextEmbedding, texts: list[str]) -> list[list[float]]:
    stacked = np.stack(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    stacked = stacked / np.maximum(norms, 1e-12)
    return stacked.tolist()


def init_vector_store() -> tuple[TextEmbedding, Any]:
    print(
        f"Loading embedding model {EMBEDDING_MODEL!r} "
        "(FastEmbed/ONNX; first run may download weights from Hugging Face)..."
    )
    embed_model = TextEmbedding(model_name=EMBEDDING_MODEL)

    chroma = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma.create_collection(
        name="rag_demo_kb_lc",
        metadata={"hnsw:space": "cosine"},
    )

    texts = [d["text"] for d in DOCUMENTS]
    ids = [d["id"] for d in DOCUMENTS]
    embeddings = _embed_normalize(embed_model, texts)
    metadatas = [{"title": d["title"]} for d in DOCUMENTS]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print("Vector store ready (Chroma in-memory, cosine index).\n")
    return embed_model, collection


def retrieve(
    embed_model: TextEmbedding,
    collection: Any,
    query: str,
    top_k: int = 2,
) -> tuple[list[dict[str, Any]], list[float]]:
    q_emb = _embed_normalize(embed_model, [query])
    result = collection.query(
        query_embeddings=q_emb,
        n_results=min(top_k, len(DOCUMENTS)),
        include=["documents", "distances", "metadatas"],
    )

    chunks: list[dict[str, Any]] = []
    distances: list[float] = []
    for i, doc_id in enumerate(result["ids"][0]):
        meta = result["metadatas"][0][i] or {}
        chunks.append(
            {
                "id": doc_id,
                "title": meta.get("title", ""),
                "text": result["documents"][0][i],
            }
        )
        distances.append(float(result["distances"][0][i]))

    return chunks, distances


def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for i, doc in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] id={doc['id']} title={doc['title']}\n{doc['text']}"
        )
    return "\n\n".join(parts)


SYSTEM_PROMPT = """
You are a helpful assistant answering from the provided CONTEXT only.

Rules:
- If the answer is not supported by CONTEXT, say you do not have that information in the retrieved documents.
- Be concise. Mention which source id you used when helpful (e.g. kb-warranty).
"""


def run_agent() -> None:
    embed_model, collection = init_vector_store()

    system = SystemMessage(content=SYSTEM_PROMPT.strip())
    history: list[HumanMessage | AIMessage] = []

    print(
        "LangChain RAG (Chroma + local embeddings + ChatGoogleGenerativeAI) — "
        "type 'exit' to quit\n"
    )

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        chunks, distances = retrieve(embed_model, collection, user_input, top_k=2)
        context_block = _format_context(chunks)

        augmented_user = (
            "CONTEXT (retrieved passages — vector similarity):\n"
            f"{context_block}\n\n"
            f"QUESTION:\n{user_input}"
        )

        messages: list[SystemMessage | HumanMessage | AIMessage] = [
            system,
            *history,
            HumanMessage(content=augmented_user),
        ]
        response = llm.invoke(messages)
        text = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        print(f"Agent: {text}\n")
        dist_str = ", ".join(f"{d:.4f}" for d in distances)
        print(
            f"(retrieved ids: {', '.join(c['id'] for c in chunks)} | "
            f"distance: [{dist_str}])\n"
        )

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=text))


if __name__ == "__main__":
    run_agent()

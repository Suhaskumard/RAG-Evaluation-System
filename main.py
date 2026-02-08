import os, json, re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

# --------- Load Document ----------
def load_doc(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

text = load_doc("data/docs.txt")

# --------- Chunking ----------
def chunk_text(text, size=500, overlap=50):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

chunks = chunk_text(text)

# --------- Embeddings ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks)

# --------- FAISS ----------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

# --------- Retrieval ----------
def expand_query(q):
    return [q, f"Explain {q}", f"How does {q} work?"]

def retrieve(query, k=10):
    results = set()
    for q in expand_query(query):
        q_emb = embedder.encode([q])[0]
        _, idx = index.search(np.array([q_emb]).astype("float32"), k)
        for i in idx[0]:
            results.add(chunks[i])
    return list(results)

# --------- Reranking ----------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs, top_k=5):
    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]

# --------- Local Generation (FLAN-T5) ----------
from transformers import pipeline

qa_model = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

def generate_answer(query, chunks):
    context = " ".join(chunks[:3])
    result = qa_model(
        question=query,
        context=context
    )
    return result["answer"]



# --------- Evaluation ----------
def faithfulness(ans, docs):
    a = embedder.encode([ans])[0]
    d = embedder.encode(docs)
    sims = np.dot(d, a) / (np.linalg.norm(d,axis=1)*np.linalg.norm(a))
    return float(np.max(sims))

def stability(query, docs):
    answers = [generate_answer(query, docs) for _ in range(3)]
    emb = embedder.encode(answers)
    sims=[]
    for i in range(len(emb)):
        for j in range(i+1,len(emb)):
            sims.append(
                np.dot(emb[i],emb[j]) /
                (np.linalg.norm(emb[i])*np.linalg.norm(emb[j]))
            )
    return float(np.mean(sims))

# --------- RUN ----------
query = "What is the goal of a RAG evaluation system?"

baseline_docs = retrieve(query, 5)
baseline_ans = generate_answer(query, baseline_docs)
baseline_faith = faithfulness(baseline_ans, baseline_docs)

retrieved = retrieve(query, 10)
reranked = rerank(query, retrieved)
final_ans = generate_answer(query, reranked)

metrics = {
    "Faithfulness": faithfulness(final_ans, reranked),
    "Stability": stability(query, reranked)
}

if metrics["Faithfulness"] < 0.6:
    with open("reports/failures.jsonl","a") as f:
        f.write(json.dumps({
            "query": query,
            "answer": final_ans,
            "metrics": metrics
        })+"\n")

print("\nFINAL ANSWER:\n", final_ans)
print("\nMETRICS:\n", metrics)
print("\nBASELINE FAITHFULNESS:", baseline_faith)

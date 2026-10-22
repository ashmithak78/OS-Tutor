from sentence_transformers import SentenceTransformer
import chromadb
import ollama
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("os_notes")
def answer_question(question):
    q_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=3)
    chunks = results["documents"][0]
    metas = results["metadatas"][0]
    context = ""
    for chunk, meta in zip(chunks, metas):
        context += f"[{meta['file']}, page {meta['page']}]: {chunk}\n\n"
    prompt = f"""You are OS-Tutor, a teaching assistant for Operating Systems.
Answer the student's question using ONLY the context below.
If the answer is not in the context, say "Not found in the notes."
Cite the file and page number(s) you used.
Context:
{context}
Question: {question}
Answer:"""
    response = ollama.generate(model="qwen2.5:3b", prompt=prompt)
    return response["response"]
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        print("\n" + answer_question(q))
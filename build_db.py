import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
DATA_FOLDER = "data"
pages = []
for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".pdf"):
        reader = PdfReader(os.path.join(DATA_FOLDER, filename))
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"file": filename, "page_number": i + 1, "text": text})

print(f"Extracted {len(pages)} pages from PDFs in '{DATA_FOLDER}'.")
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
all_chunks, all_metadata, all_ids = [], [], []
chunk_id = 0
for page in pages:
    for chunk in chunk_text(page["text"]):
        all_chunks.append(chunk)
        all_metadata.append({"file": page["file"], "page": page["page_number"]})
        all_ids.append(str(chunk_id))
        chunk_id += 1
print(f"Created {len(all_chunks)} chunks.")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(all_chunks).tolist()
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("os_notes")
collection.add(ids=all_ids, documents=all_chunks, embeddings=embeddings, metadatas=all_metadata)
print("Done. Knowledge base saved in the 'chroma_db' folder.")
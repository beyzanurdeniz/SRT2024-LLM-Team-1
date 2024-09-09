import os
import pickle
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from local import gemini_api_key, pinecone_api_key, root_path

pdf_files = [
    f"{root_path}/cphb.pdf",
    f"{root_path}/inzva-bundles/01-intro.pdf",
    f"{root_path}/inzva-bundles/02-algorithms-1.pdf",
    f"{root_path}/inzva-bundles/03_math1.pdf",
    f"{root_path}/inzva-bundles/04_Graph1.pdf",
    f"{root_path}/inzva-bundles/05_DP1.pdf",
    f"{root_path}/inzva-bundles/06-data-structures-1.pdf",
    f"{root_path}/inzva-bundles/07_Graph2.pdf",
    f"{root_path}/inzva-bundles/08-data-structure-2.pdf",
    f"{root_path}/inzva-bundles/10_dp_2.pdf",
    f"{root_path}/inzva-bundles/11-graph-3.pdf",
    f"{root_path}/inzva-bundles/12_Math3.pdf",
    f"{root_path}/inzva-bundles/14-algorithms-5.pdf",
    f"{root_path}/inzva-bundles/Data-Structures-3.pdf"
]

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index_name = "cp-index"

# Initialize Gemini model
gemini_model = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-1.5-flash")

# Check if index exists
def ensure_index_exists():
    index_list = pc.list_indexes()
    flag = False
    for index in index_list:
        if index.name == index_name:
            flag = True
            break
    if not flag:
        pc.create_index(name=index_name, dimension=384, metric="cosine", shards=1, timeout=600)

ensure_index_exists()
index = pc.Index(index_name)

# Process PDF and split text into chunks
def process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(data)
        texts = [str(doc) for doc in documents]
        return texts
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

# Load tokenizer and model for embedding generation
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Generate embedding for a given text
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy().tolist()[0]


# Upload text chunks and their embeddings to Pinecone
def upload_embeddings_to_pinecone(text_chunks):
    for i, chunk in enumerate(text_chunks):
        try:
            embedding = generate_embedding(chunk)
            index.upsert([(f"doc_{i}", embedding)])
        except Exception as e:
            print(f"Error embedding text chunk {i}: {e}")

# Process PDF files if they haven't been processed already
def process_and_upload_pdfs():
    global all_text_chunks
    if not os.path.exists("processed_text_chunks.pkl"):
        all_text_chunks = []
        for pdf_file in pdf_files:
            all_text_chunks.extend(process_pdf(pdf_file))
        
        # Save processed chunks
        with open("processed_text_chunks.pkl", "wb") as f:
            pickle.dump(all_text_chunks, f)

        # Upload embeddings to Pinecone
        upload_embeddings_to_pinecone(all_text_chunks)
        print("PDFs processed and embeddings uploaded to Pinecone.")
    else:
        # Load previously processed chunks
        with open("processed_text_chunks.pkl", "rb") as f:
            all_text_chunks = pickle.load(f)
        
        # Check if Pinecone index contains vectors
        try:
            search_result = index.query(vector=generate_embedding("test"), top_k=1)
            if len(search_result['matches']) == 0:
                print("Re-uploading embeddings to Pinecone...")
                upload_embeddings_to_pinecone(all_text_chunks)
            else:
                print("Embeddings already in Pinecone. Ready for querying.")
        except Exception as e:
            print(f"Error querying Pinecone for check: {e}")
            upload_embeddings_to_pinecone(all_text_chunks)

process_and_upload_pdfs()

# Generate response based on retrieved document chunks
def generate_response_from_docs(document_ids):
    relevant_texts = []
    for doc_id in document_ids:
        try:
            doc_index = int(doc_id.split("_")[1])
            if doc_index < len(all_text_chunks):
                relevant_texts.append(all_text_chunks[doc_index])
        except Exception as e:
            print(f"Error retrieving document for {doc_id}: {e}")
    combined_text = " ".join(relevant_texts)
    response = gemini_model.invoke(f"Based on the following information: {combined_text}, answer the question. Don't include any invisible code snippets. Just explanations, or if you are going to add a code, do it visible.")
    return response.content

# Query Pinecone with a user query
def query_pinecone(query, relevance_threshold=0.3):  # Define a threshold for relevance
    try:
        query_embedding = generate_embedding(query)
        search_result = index.query(vector=query_embedding, top_k=3)
        
        matches = search_result['matches']
        document_ids = []
        distances = []

        # Filter based on relevance threshold
        for match in matches:
            if match['score'] > relevance_threshold:  # Adjust based on your desired relevance level
                document_ids.append(match['id'])
                distances.append(match['score'])
        
        return document_ids, distances
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return [], []

# Main program loop
print("Welcome to the Competitive Programming Helper.")

while True:
    user_query = input("Enter your question (or 'exit' to quit): ")
    if user_query.lower() == "exit":
        break

    document_ids, distances = query_pinecone(user_query)
    
    if document_ids:
        answer = generate_response_from_docs(document_ids)
        print(f"Answer: {answer}")
    else:
        print("No relevant information found.")


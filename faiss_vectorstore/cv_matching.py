from sklearn.cluster import KMeans
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
import json
import re
import os

# Initialize SentenceTransformer for Embeddings
embedding_model = SentenceTransformer('msmarco-distilbert-base-v3')

# Initialize LLaMA LLM for extracting key information
llm = Ollama(model="llama3.1:8b")

# File paths for saved embeddings and metadata
embedding_file_path = "job_embeddings.npy"
metadata_file_path = "job_metadata.json"
faiss_index_file_path = "faiss_index.bin"

# Number of clusters to create
NUM_CLUSTERS = 5

def extract_json_from_text(response):
    """
    Extract JSON from text response. It handles any JSON enclosed in ```json``` blocks.
    """
    try:
        # Search for JSON block enclosed in ```json ... ```
        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if json_match:
            # Load JSON from matched block
            return json.loads(json_match.group(1))
        else:
            return {"error": "No JSON object found in response"}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to decode JSON: {str(e)}"}

def parse_and_extract_cv(pdf_path):
    """
    Load and extract text from the CV PDF file, then use LLM to extract key information.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    cv_text = "\n".join([doc.page_content for doc in documents])
    print("CV text extract bhayo")
    print(cv_text)
    # Use LLM to extract structured data
    prompt = f"""
    Extract the following structured data from this resume text:
    - Basics: Name, Email
    - Work Experience: List each company, position, years of experience, and a brief description of responsibilities and achievements
    - Education: List each institution, degree, and field of study
    - Skills: List all skills mentioned
    - Projects: For each project, include the title, description, and technologies used

    Resume text:
    {cv_text}

    Respond with a JSON object matching the structure:
    {{
        "basics": {{"name": "", "email": ""}},
        "work experience": [{{"company": "", "position": "", "years": "", "description": ""}}],
        "education": [{{"institution": "", "degree": ""}}],
        "skills": [],
        "projects": [{{"title": "", "description": "", "technologies": []}}]
    }}
    """
    response = llm.invoke(prompt)
    print(response)
    try:
        extracted_data = extract_json_from_text(response)
        print("Extracted data")
        print(extracted_data)
    except (json.JSONDecodeError, AttributeError):
        extracted_data = {"error": "Failed to parse LLM response"}
    return cv_text, extracted_data

def create_faiss_store_with_clustering(file_path):
    """
    Create or load a FAISS vector store with clustering from job descriptions in a CSV file.
    """
    if os.path.exists(embedding_file_path) and os.path.exists(metadata_file_path) and os.path.exists(faiss_index_file_path):
        # Load existing FAISS index, embeddings, and metadata
        index = faiss.read_index(faiss_index_file_path)
        embeddings = np.load(embedding_file_path)
        with open(metadata_file_path, "r") as f:
            metadata = json.load(f)
        print("Loaded embeddings, metadata, and FAISS index from files.")

        # Reload KMeans for clustering
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
        kmeans.fit(embeddings)  # Refit using saved embeddings for consistency
    else:
        # Create new FAISS index
        index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        job_descriptions = pd.read_csv(file_path)

        all_embeddings = []
        metadata = []

        for _, row in job_descriptions.iterrows():
            content = f"{row['Job Title']}\n{row['Job Description']}"
            embedding = embedding_model.encode(content)
            all_embeddings.append(embedding)
            metadata.append({"job_title": row['Job Title'], "job_description": row['Job Description']})

        # Convert embeddings to NumPy array
        all_embeddings = np.array(all_embeddings).astype(np.float32)

        # Cluster embeddings
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
        cluster_assignments = kmeans.fit_predict(all_embeddings)

        # Add cluster information to metadata
        for i, cluster in enumerate(cluster_assignments):
            metadata[i]["cluster"] = int(cluster)

        # Add embeddings to FAISS index
        index.add(all_embeddings)

        # Save embeddings, metadata, and FAISS index
        np.save(embedding_file_path, all_embeddings)
        with open(metadata_file_path, "w") as f:
            json.dump(metadata, f)
        faiss.write_index(index, faiss_index_file_path)

        print("Generated and saved embeddings, metadata, and FAISS index with clustering.")

    return index, metadata, kmeans

def retrieve_top_jobs_within_cluster(cv_embedding, cluster_metadata, k=4):
    """
    Retrieve the top `k` jobs within the nearest cluster based on similarity scores.
    """
    cluster_embeddings = [
        embedding_model.encode(f"{m['job_title']}\n{m['job_description']}").astype(np.float32)
        for m in cluster_metadata
    ]

    cluster_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    cluster_index.add(np.array(cluster_embeddings))

    distances, indices = cluster_index.search(np.array([cv_embedding]), k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        similarity = float(1 / (1 + dist))  # Convert distance to similarity and ensure type is `float`
        results.append({
            "job_title": cluster_metadata[idx]["job_title"],
            "job_description": cluster_metadata[idx]["job_description"],
            "similarity_score": similarity
        })

    return results


def process_new_resume(cv_text, metadata, kmeans, k=4):
    """
    Process a new resume to find the nearest cluster and retrieve top jobs.
    """
    # Compute embedding for the new resume
    cv_embedding = embedding_model.encode(cv_text).astype(np.float32)

    # Find the nearest cluster
    cluster_distances = np.linalg.norm(kmeans.cluster_centers_ - cv_embedding, axis=1)
    closest_cluster = np.argmin(cluster_distances)
    print(f"Nearest Cluster: {closest_cluster}")

    # Filter metadata for jobs in the nearest cluster
    cluster_metadata = [m for m in metadata if m.get("cluster") == closest_cluster]
    print(f"Number of jobs in cluster {closest_cluster}: {len(cluster_metadata)}")

    # Retrieve top jobs within the cluster
    top_jobs = retrieve_top_jobs_within_cluster(cv_embedding, cluster_metadata, k=k)
    return top_jobs


def main():
    # Path to the job descriptions CSV file
    job_file_path = "../job_title_des.csv"

    # Path to the candidate CV PDF
    cv_path = "../Nadika_Poudel_CV__.pdf"

    # Parse and Extract CV
    cv_text, extracted_data = parse_and_extract_cv(cv_path)
    print("Extracted Data from CV:", json.dumps(extracted_data, indent=2))

    # Create FAISS Store with Clustering
    index, metadata, kmeans = create_faiss_store_with_clustering(job_file_path)

    # Process the new resume
    top_jobs = process_new_resume(cv_text, metadata, kmeans)

    # Save Recommendations to JSON
    with open("recommendations.json", "w") as f:
        json.dump(top_jobs, f, indent=2)

    # Output Results
    print(json.dumps(top_jobs, indent=2))

if __name__ == "__main__":
    main()

import psycopg
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
import re
import json
import os


# PostgreSQL Configuration
DB_CONFIG = {
    'dbname': 'jobins',
    'user': 'nadika',
    'password': 'nadika',  
    'host': '127.0.0.1',
    'port': 5432
}

# Initialize SentenceTransformer for Embedding
embedding_model = SentenceTransformer('msmarco-distilbert-base-v3')


# Initialize LLaMA LLM for extracting key information
llm = Ollama(model="llama3.1:8b")

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

async def parse_and_extract_cv(pdf_path):
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
        extracted_data = await extract_json_from_text(response)
        print("Extracted data")
        print(extracted_data)
    except (json.JSONDecodeError, AttributeError):
        extracted_data = {"error": "Failed to parse LLM response"}
    return extracted_data


# Convert dummy CV to a string and generate a vector
def generate_cv_vector(cv_data):
    cv_text = (
        f"{cv_data['basics']['name']}\n"
        f"{cv_data['basics']['email']}\n"
        f"{' '.join(cv_data['skills'])}\n"
        + "\n".join(
            f"{exp['position']} at {exp['company']} - {exp['description']}"
            for exp in cv_data['work experience']
        )
        + "\n".join(
            f"{proj['title']} - {proj['description']} (Technologies: {', '.join(proj['technologies'])})"
            for proj in cv_data['projects']
        )
    )
    return embedding_model.encode(cv_text).tolist()

# Find top similar jobs for a CV
def find_top_similar_jobs_for_cv(cv_vector, top_n=5):
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            # Perform similarity search
            cursor.execute("""
                SELECT id, job_title, job_description, 
                       1 - (embedding <=> %s::vector) AS similarity
                FROM job_embeddings
                ORDER BY similarity DESC
                LIMIT %s;
            """, (cv_vector, top_n))
            results = cursor.fetchall()

    # Format results as JSON
    top_jobs = [
        {
            "job_id": result[0],
            "job_title": result[1],
            "job_description": result[2],
            "similarity_score": float(result[3])  # Convert similarity to float for JSON serialization
        }
        for result in results
    ]

    # Display results in the console
    print("Top Jobs for the CV:")
    print(json.dumps(top_jobs, indent=2))

    # Save results to recommendations.json
    with open("recommendations.json", "w") as f:
        json.dump(top_jobs, f, indent=2)

    return top_jobs

def main():
     # Path to the candidate CV PDF
    cv_path = "../Nadika_Poudel_CV__.pdf"

    # Get dummy CV
    cv_data = parse_and_extract_cv(cv_path)

    # Generate CV vector
    cv_vector = generate_cv_vector(cv_data)

    # Find top similar jobs for the CV and display as JSON
    find_top_similar_jobs_for_cv(cv_vector, top_n=5)

if __name__ == "__main__":
    main()

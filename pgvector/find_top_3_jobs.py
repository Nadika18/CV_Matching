import psycopg
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# PostgreSQL Configuration
DB_CONFIG = {
    'dbname': 'jobins',
    'user': 'nadika',
    'password': 'nadika',  # Replace with your actual password
    'host': '127.0.0.1',
    'port': 5432
}

# Initialize SentenceTransformer for Embedding
embedding_model = SentenceTransformer('msmarco-distilbert-base-v3')

# Dummy CV data for a Machine Learning Engineer
def get_dummy_cv():
    return {
        "basics": {
            "name": "John Doe",
            "email": "johndoe@example.com"
        },
        "work_experience": [
            {
                "company": "TechCorp",
                "position": "Machine Learning Engineer",
                "years": "3 years",
                "description": "Developed and deployed machine learning models for predictive analytics."
            },
            {
                "company": "AI Solutions",
                "position": "Data Scientist",
                "years": "2 years",
                "description": "Performed data preprocessing and feature engineering for AI projects."
            }
        ],
        "skills": [
            "Python", "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch"
        ],
        "projects": [
            {
                "title": "Fraud Detection System",
                "description": "Built a machine learning model to detect fraudulent transactions.",
                "technologies": ["Python", "TensorFlow", "SQL"]
            }
        ]
    }

# Convert dummy CV to a string and generate a vector
def generate_cv_vector(dummy_cv):
    cv_text = (
        f"{dummy_cv['basics']['name']}\n"
        f"{dummy_cv['basics']['email']}\n"
        f"{' '.join(dummy_cv['skills'])}\n"
        + "\n".join(
            f"{exp['position']} at {exp['company']} - {exp['description']}"
            for exp in dummy_cv['work_experience']
        )
        + "\n".join(
            f"{proj['title']} - {proj['description']} (Technologies: {', '.join(proj['technologies'])})"
            for proj in dummy_cv['projects']
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
    # Get dummy CV
    dummy_cv = get_dummy_cv()

    # Generate CV vector
    cv_vector = generate_cv_vector(dummy_cv)

    # Find top similar jobs for the CV and display as JSON
    find_top_similar_jobs_for_cv(cv_vector, top_n=5)

if __name__ == "__main__":
    main()

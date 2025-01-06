import psycopg
import numpy as np
import json
from sentence_transformers import SentenceTransformer

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

# Read CV data from JSON file
def read_cv_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Save recommendations to JSON file
def save_recommendations_to_json(recommendations, file_path):
    with open(file_path, 'w') as file:
        json.dump(recommendations, file, indent=4)

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

# Find top 5 similar jobs for a CV
def find_top_similar_jobs_for_cv(cv_vector):
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            # Perform similarity search
            cursor.execute("""
                SELECT id, job_title, job_description, 
                       1 - (embedding <=> %s::vector) AS similarity
                FROM jobs_without_clustering
                ORDER BY similarity DESC
                LIMIT 5;
            """, (cv_vector,))
            results = cursor.fetchall()

    recommendations = []
    for result in results:
        recommendations.append({
            "id": result[0],
            "title": result[1],
            "similarity": round(result[3], 4),
            "description": result[2]
        })
    
    return recommendations

def main():
    # Read CV data from JSON file
    dummy_cv = read_cv_from_json('CV3.json')

    # Generate CV vector
    cv_vector = generate_cv_vector(dummy_cv)

    # Find top 5 similar jobs for the CV
    recommendations = find_top_similar_jobs_for_cv(cv_vector)

    # Save recommendations to JSON file
    save_recommendations_to_json(recommendations, 'recommendations3.json')

    print("Recommendations have been saved to recommendations3.json")

if __name__ == "__main__":
    main()

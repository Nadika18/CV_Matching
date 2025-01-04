from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import psycopg
import os
import ast  # Import for safely parsing string representations


# Initialize SentenceTransformer for Embeddings
embedding_model = SentenceTransformer('msmarco-distilbert-base-v3')

# PostgreSQL Configuration
DB_CONFIG = {
    'dbname': 'jobins',
    'user': 'nadika',
    'password': 'nadika',
    'host': 'localhost',
    'port': 5432
}

# Table names
EMBEDDING_TABLE = "job_embeddings"
CLUSTER_TABLE = "cluster_centroids"


def setup_pgvector():
    """
    Ensure the PostgreSQL database is ready with the required tables for storing embeddings and cluster centroids.
    """
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            # Create pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create the table for embeddings and metadata
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {EMBEDDING_TABLE} (
                    id SERIAL PRIMARY KEY,
                    job_title TEXT,
                    job_description TEXT,
                    embedding VECTOR(768), -- Embedding stored as VECTOR type
                    cluster INTEGER
                );
            """)
            
            # Create the table for cluster centroids
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {CLUSTER_TABLE} (
                    id SERIAL PRIMARY KEY,
                    centroid VECTOR(768) -- Centroid stored as VECTOR type
                );
            """)
            conn.commit()
    print(f"Tables `{EMBEDDING_TABLE}` and `{CLUSTER_TABLE}` are ready.")


def store_embeddings_in_pg(job_descriptions):
    """
    Store job descriptions and their embeddings in PostgreSQL.
    """
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            for _, row in job_descriptions.iterrows():
                content = f"{row['Job Title']}\n{row['Job Description']}"
                embedding = embedding_model.encode(content).tolist()
                cursor.execute(f"""
                    INSERT INTO {EMBEDDING_TABLE} (job_title, job_description, embedding, cluster)
                    VALUES (%s, %s, %s, NULL);
                """, (row['Job Title'], row['Job Description'], embedding))
            conn.commit()
    print("Embeddings and metadata stored in PostgreSQL.")


def retrieve_embeddings():
    """
    Retrieve all embeddings and metadata from PostgreSQL.
    """
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT id, embedding FROM {EMBEDDING_TABLE};")
            results = cursor.fetchall()

    ids = [record[0] for record in results]
    # Parse the string representation of the vector into a NumPy array
    embeddings = np.array([np.array(ast.literal_eval(record[1]), dtype=np.float32) for record in results])
    return ids, embeddings



def store_cluster_centroids(kmeans):
    """
    Store cluster centroids in PostgreSQL.
    """
    centroids = kmeans.cluster_centers_.tolist()
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"TRUNCATE TABLE {CLUSTER_TABLE};")
            for centroid in centroids:
                cursor.execute(f"""
                    INSERT INTO {CLUSTER_TABLE} (centroid)
                    VALUES (%s);
                """, (centroid,))
            conn.commit()
    print("Cluster centroids stored in PostgreSQL.")


def update_clusters_in_pg(ids, cluster_assignments):
    """
    Update cluster assignments in PostgreSQL.
    """
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            for job_id, cluster in zip(ids, cluster_assignments):
                cursor.execute(f"""
                    UPDATE {EMBEDDING_TABLE}
                    SET cluster = %s
                    WHERE id = %s;
                """, (int(cluster), job_id))
            conn.commit()
    print("Cluster assignments updated in PostgreSQL.")


def perform_clustering():
    """
    Perform clustering on the embeddings and update the database with cluster assignments and centroids.
    """
    # Retrieve embeddings from PostgreSQL
    ids, embeddings = retrieve_embeddings()

    # Perform KMeans clustering
    optimal_clusters = 3  # Fixed for simplicity; can use elbow method for dynamic calculation
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(embeddings)

    # Update clusters in the database
    update_clusters_in_pg(ids, cluster_assignments)

    # Store cluster centroids
    store_cluster_centroids(kmeans)


def main():
    # Path to the job descriptions CSV file
    job_file_path = "../job_title_des.csv"

    # Set up PostgreSQL tables
    setup_pgvector()

    # Load job descriptions
    job_descriptions = pd.read_csv(job_file_path)

    # Store embeddings and metadata in PostgreSQL
    store_embeddings_in_pg(job_descriptions)

    # Perform clustering
    perform_clustering()


if __name__ == "__main__":
    main()

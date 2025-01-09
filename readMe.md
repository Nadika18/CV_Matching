# Job Vacancy Recommendation System

#### Framework used:
- Langchain

#### LLM :
- Llama-3.1-8B (locally using Ollama)

#### Embedding Model:
- Sentence Transformer - msmarco-distilbert-base-v3  
  [Link in Hugging Face](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v3)

#### Vectorstore:
Implemented the system in two different ways with different vector-store used.
- pgvector with database on Postgres SQL
- FAISS vector-store

#### Input: 
- Pdf file of CV
- Nadika_Poudel_CV__.pdf (Example above )

#### Output:
- Json file with top n recommendations. One on the top is highly relevant than others. 
- faiss_vectorstore/recommendations.json  ( Example above)

## Setup


- Setup postgres SQL and pgvector using docker file.
```bash
docker compose up -d
```

- Install other requirements using requirements.txt

```bash
pip install -r requirements.txt
```



- ### Setting up Ollama
1. Download and Install Ollama
Visit Ollama's download page https://ollama.com/download and follow the installation instructions specific to your operating system (Windows, macOS, or Linux). 

2. Pull Required Models
To use the LLaMA models for auto-evaluation, pull them locally by running the following commands:

```bash
ollama run llama3.1
```


3. Start the Ollama Server
After pulling the models, start the Ollama server to serve the models:

```bash
ollama serve 
```

**Note: Make sure that Ollama runs in background while running the app**


- ### Running the app
    - Run implementation with pgvector:
    ```bash
    cd pgvector
    python3 add_embeddings.py
    python3 find_top_jobs.py
    ```

    - Run implementation with Faiss vector store:
    ```bash
    cd faiss_vectorstore
    python3 cv_matching.py
    ```

        
## Implementation Details

### CV Parsing
- Extracted the text from PDF using Langchain's PyPDFLoader.
- Extracted text is passed to LLM(Llama 3.1 ) for extracting the key features in json format given below:

```json
{
    "basics": {
        "name": "",
        "email": ""
    },
    "work experience": [
        {
            "company": "",
            "position": "",
            "years": "",
            "description": ""
        }
    ],
    "education": [
        {
            "institution": "",
            "degree": ""
        }
    ],
    "skills": [],
    "projects": [
        {
            "title": "",
            "description": "",
            "technologies": []
        }
    ]
}
```



### Recommendation system

- Implemented using cosine similarity between job description and job vacancy details.
- Top n jobs with the highest cosine similarity with the CV's details is retrieved.

### Job Clustering

- Job clustering is implemented using K-Nearest Neighbout (KNN) algorithm.
- Job vacancy details with the similar embedding vectors are kept in single cluster.
- CV details with the highest cosine similarity among cluster's center is fetched first. Then, the top n jobs with the highest cosine similarity if fetched on that cluster.

## Implementation 1: With FAISS vector store

<p align="center">
  <img src="images/FAISS.png" alt="Implementation with FAISS vector store details">
</p>

- **Probable Issue:** If the semantic similarity with the cluster's center is misleading, the recommendation system fails. So,need to verify that in future work.

## Implementation 2: With pgvector store and Postgres SQL

<p align="center">
  <img src="images/pgvector.png" alt="Implementation with pgvector ">
</p>

- **Current Issues:** Recommendation with the clustering didnot work well. So, need to improve it. 

## File Descriptions
1. faiss_vectorstore : Includes files for setup with FAISS vectorstore
    - cv_matching.py : Store job vacancy with its embeddings if not present. Else fetch the stored embeddings. Parse the CV and give top n matched job vacancies.

    - recommendations.json : Top n jobs matched for CV file( Nadika_Poudel_CV__.pdf)

2. pgvector: Includes files for setup with pgvector and Postgres SQL
    - add_embeddings.py : Store initial jobs and its embeddings in postgres.
    - find_top_jobs.py : Fetch matched top n jobs.
    - recommendations.json : Top n jobs matched for CV file(Nadika_Poudel_CV__.pdf)
    - find_top_jobs_without_clustering.py : Since, running LLM locally on Ollama takes a lot of time. Here, recommendation system can be tested passing CV key details on json file.
    -CV2.json and CV3.json : Sample key details of CV.
    - recommendations2.json and recommendations3.json : Job recommendations fetched for CV2.json and CV3.json respectively.

3. job_title_desc.csv : Dataset of job vacanies with job_title and job_description.

4. Nadika_Poudel_CV__.pdf : Example CV file tested with. 

## Future work:
- Generate test dataset and check the relevancy of the CV with the job vancancy. This will assess the performance of recommendation system.
- Implement hierarchical clustering system.


## Final notes:
Currently, the implementation with the FAISS vector-store is fetching good results. But the implementation with the pgvector sometimes retrieves misleading results.
Vertex AI Embedding Retrieval Demo (BEIR + FAISS)

This project demonstrates how to use Google Cloud Vertex AI text embeddings together with the BEIR benchmark datasets and FAISS to build and evaluate a simple retrieval system in Python.

The example uses the NFCorpus dataset to:

    Download documents and queries via BEIR.

    Compute embeddings with Vertex AI text-embedding-005.

    Index document embeddings using FAISS (L2 index).

    Retrieve nearest neighbors for sample queries.

    Evaluate retrieval quality with pytrec_eval (e.g., NDCG@10, P@1, Recall@10).

Requirements

    Python 3.9+ (tested with recent 3.x).

    A Google Cloud project with Vertex AI enabled and billing active.

    Credentials for Google Cloud (e.g., Application Default Credentials via gcloud auth application-default login).

    macOS on Intel or Apple Silicon (works on Linux as well with the same steps).

Python dependencies (installed via pip or a virtualenv):

    google-cloud-aiplatform (Vertex AI SDK; provides vertexai and TextEmbeddingModel)

    beir

    faiss-cpu

    numpy

    pandas

    pytrec_eval

Example install:

bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install \
  google-cloud-aiplatform \
  beir \
  faiss-cpu \
  numpy \
  pandas \
  pytrec_eval

If faiss-cpu fails on Apple Silicon, use a Conda environment or platform-specific build; on Intel macOS it usually installs directly from PyPI.
Google Cloud Setup

    Install the Google Cloud CLI and authenticate:

bash
# If using Homebrew on macOS:
brew install --cask google-cloud-sdk

gcloud init
gcloud auth application-default login

    Enable Vertex AI in your project:

bash
gcloud services enable aiplatform.googleapis.com

    Note your:

    Project ID (e.g., google-cloud-project-id)

    Region where embeddings are available (e.g., europe-west4)

Update the script:

python
vertexai.init(project="YOUR_PROJECT_ID", location="YOUR_REGION")

Script Overview

The core code:

python
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import faiss
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import numpy as np
import pandas as pd
import pytrec_eval
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

Embedding function

python
def embed_text(texts, model, task, batch_size=5):
    embed_mat = np.zeros((len(texts), 768))
    for batch_start in range(0, len(texts), batch_size):
        size = min(len(texts) - batch_start, batch_size)
        inputs = [
            TextEmbeddingInput(texts[batch_start + i], task_type=task)
            for i in range(size)
        ]
        embeddings = model.get_embeddings(inputs)
        for i in range(size):
            embed_mat[batch_start + i, :] = embeddings[i].values
    return embed_mat

    Embeds a list of strings into a dimensional numpy array using the chosen Vertex AI model.

    task is either "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY".

Data loading (BEIR NFCorpus)

python
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
data_path = util.download_and_unzip(url, "datasets")

corpus, queries, qrels = GenericDataLoader("datasets/nfcorpus").load(split="test")

    Downloads and unzips NFCorpus into datasets/.

    Loads:

        corpus: document  document dict (including 'text')

        queries: query  query text

        qrels: query  relevant document IDs with relevance scores

Vertex AI model initialization

python
vertexai.init(project="google-cloud-project-id", location="europe-west4")
model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    Initializes Vertex AI for the specified project and region.

    Loads the text-embedding-005 model.

Embedding documents and building FAISS index

python
doc_ids, docs = zip(*[(doc_id, doc['text']) for doc_id, doc in corpus.items()])

doc_embeddings = embed_text(docs, model, "RETRIEVAL_DOCUMENT")
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

    Extracts document texts and IDs.

    Embeds all documents as "RETRIEVAL_DOCUMENT".

    Builds a FAISS L2 index and adds all document embeddings.

Sample queries

python
example_embed = embed_text(['Lactose intolerance curable?'],
                           model, 'RETRIEVAL_QUERY')
s, q = index.search(example_embed, 1)
print(f'Score: {s[0][0]:.2f}, Text: "{docs[q[0][0]]}"')

example_embed = embed_text(['Is Caffeinated Tea Really Dehydrating?'],
                           model, 'RETRIEVAL_QUERY')
s, q = index.search(example_embed, 1)
print(f'Score: {s[0][0]:.2f}, Text: "{docs[q[0][0]]}"')

    Embeds example questions as "RETRIEVAL_QUERY".

    Searches for the nearest document and prints score and text.

Full evaluation with pytrec_eval

python
q_ids, questions = zip(*[(q_id, q) for q_id, q in queries.items()])
query_embeddings = embed_text(questions, model, "RETRIEVAL_QUERY")
q_scores, q_doc_ids = index.search(query_embeddings, 10)

    Embeds all queries and searches top-10 documents for each query.

python
search_qrels = {
    q_ids[i]: {
        doc_ids[_id]: -1 * s.item()
        for _id, s in zip(q_doc_ids[i], q_scores[i])
    }
    for i in range(len(q_ids))
}

    Builds a   score dict for pytrec_eval, multiplying scores by -1 because FAISS uses distances (smaller is better).

python
evaluator = pytrec_eval.RelevanceEvaluator(
    qrels, {'ndcg_cut.10', 'P_1', 'recall_10'}
)
eval_results = evaluator.evaluate(search_qrels)

df = pd.DataFrame.from_dict(eval_results, orient='index')
print(df.mean())

    Evaluates metrics NDCG@10, Precision@1, Recall@10.

    Prints mean metrics across all queries.

    Example typical values:

        P_ 0.52 (precision@1)

        recall_ 0.20 (recall@10)

Running the Script

    Ensure your virtualenv is active and dependencies installed.

    Set Google Cloud credentials (ADC) as described in the Google Cloud Setup section.

    Run:

bash
python vertex_beir_faiss_demo.py

Where vertex_beir_faiss_demo.py contains the code above.
Notes for Intel vs Apple Silicon

    The Python code is the same on both architectures.

    Main differences are in:

        Installing FAISS (faiss-cpu) and other native libs.

        Ensuring OpenMP / MKL libs do not conflict; the script sets:

        python
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    to avoid Intel MKL duplicate library errors.

    If you hit architecture-specific issues on Apple Silicon:

        Try a Conda environment (e.g., conda install faiss-cpu -c pytorch).

        Avoid mixing Rosetta and native ARM Python in the same environment.


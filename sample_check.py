from beir import util
from beir.datasets.data_loader import GenericDataLoader
import faiss
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import numpy as np
import pandas as pd
import pytrec_eval

x = np.random.rand(100, 768).astype('float32')
index = faiss.IndexFlatL2(768)
index.add(x)
d, i = index.search(x[:1], 5)
print(d, i)
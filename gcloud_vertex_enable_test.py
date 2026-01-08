import os
import google.auth
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Method 1: Environment variable (simplest)
os.environ["GOOGLE_CLOUD_QUOTA_PROJECT_ID"] = "kaggle-genai-chap2-embed"

# Initialize (quota project now picked up automatically)
vertexai.init(project="kaggle-genai-chap2-embed", location="europe-west4")

print("✅ Initialization successful!")

# Test model load
model = TextEmbeddingModel.from_pretrained("text-embedding-005")
print("✅ Model loaded successfully!")

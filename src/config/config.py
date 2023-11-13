import os
from pathlib import Path

from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from transformers import AutoTokenizer
from utilities.custom_logger import CustomLogger

logger = CustomLogger()

# set paths
ROOT = Path(__file__).parents[2]
SRC = os.path.join(ROOT,"src")
MODEL_PATH = os.path.join(SRC , "model", "ggml-model-q4_k_m.gguf") 
EMBED_PATH = "/Users/bo/workspace/KR-SBERT-V40K-klueNLI-augSTS"
TOKENIZER = AutoTokenizer.from_pretrained(EMBED_PATH)

# enable in-memory caching
set_llm_cache(InMemoryCache())

CHUNK_SIZE = 512
CLUSTERING_MAX = 11
RESOLUTION_MAX = 2.0
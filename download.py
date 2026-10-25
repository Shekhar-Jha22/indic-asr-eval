"""Fetch FLEURS hi_in parquet shards into data/ (~1.9 GB)."""
from huggingface_hub import hf_hub_download

for split in ("train", "validation", "test"):
    hf_hub_download("google/fleurs", f"parquet-data/hi_in/{split}-00000-of-00001.parquet",
                    repo_type="dataset", local_dir="data")

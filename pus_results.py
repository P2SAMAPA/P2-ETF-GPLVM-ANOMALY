from huggingface_hub import HfApi
from pathlib import Path
import config

def push_daily_result(local_path: Path):
    repo_id = config.OUTPUT_REPO
    token = config.HF_TOKEN
    if not token:
        print("No HF_TOKEN, skipping upload")
        return
    api = HfApi(token=token)
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"Uploaded to {repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")

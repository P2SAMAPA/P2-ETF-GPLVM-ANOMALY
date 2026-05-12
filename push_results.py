"""
Upload results to Hugging Face – automatically creates a new dataset repository if needed.
"""
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import config
import os

def push_daily_result(local_path: Path):
    token = config.HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ No HF_TOKEN found. Skipping upload.")
        return

    # Use a dedicated repo name for this engine
    repo_id = "P2SAMAPA/p2-etf-gplvm-anomaly-results"

    api = HfApi(token=token)

    # Check if repository exists; if not, create it
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"✅ Repository '{repo_id}' already exists.")
    except Exception:
        print(f"🆕 Creating repository '{repo_id}'...")
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", private=False, token=token)
            print(f"✅ Created repository '{repo_id}'.")
        except Exception as e:
            print(f"❌ Failed to create repo: {e}")
            return

    # Upload the file
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {local_path.name}"
        )
        print(f"✅ Uploaded {local_path.name} to {repo_id}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

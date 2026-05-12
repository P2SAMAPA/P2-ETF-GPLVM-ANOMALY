"""
Upload results to Hugging Face dataset repository using explicit commit.
"""
from huggingface_hub import HfApi
from pathlib import Path
import config
import os

def push_daily_result(local_path: Path):
    token = config.HF_TOKEN or os.environ.get("HF_TOKEN")
    repo_id = config.OUTPUT_REPO
    if not token:
        print("❌ No HF_TOKEN found. Skipping upload.")
        return

    api = HfApi(token=token)
    try:
        # The most reliable method: upload_file with explicit commit_message
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
        print("   Trying alternative method...")
        # Fallback: use requests as before
        import requests
        url = f"https://huggingface.co/api/datasets/{repo_id}/upload/{local_path.name}"
        headers = {"Authorization": f"Bearer {token}"}
        with open(local_path, "rb") as f:
            files = {"file": (local_path.name, f, "application/json")}
            response = requests.post(url, headers=headers, files=files)
        if response.status_code == 200:
            print(f"✅ Uploaded via fallback: {local_path.name}")
        else:
            print(f"❌ Fallback also failed: {response.status_code} {response.text}")

"""Fetch only the private, inference-only asset repository."""
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def ensure_assets():
    local = Path(os.getenv('ASSETS_DIR', str(Path(__file__).resolve().parents[1]/'assets')))
    if (local/'recommender.pt').is_file():
        return local
    repo = os.getenv('ASSET_REPO_ID')
    if not repo:
        raise RuntimeError('Set ASSET_REPO_ID or provide the inference assets directory.')
    return Path(snapshot_download(repo_id=repo, token=os.getenv('HF_TOKEN'),
        revision=os.getenv('ASSET_REVISION', 'main'),
        allow_patterns=['*.pt','*.json','*.npz','*.npy','sentence-t5-base/*']))

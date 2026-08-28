"""
Script utilitário para download dos modelos pré-treinados e datasets do AnyGraph via Hugging Face.
Uso:
    python download_assets.py --models      # Baixa os modelos pré-treinados (~9.5 GB)
    python download_assets.py --datasets    # Baixa os datasets zero-shot (~4 GB)
    python download_assets.py --all         # Baixa modelos e datasets
"""

import os
import sys
import tarfile
import zipfile
import argparse
from huggingface_hub import hf_hub_download

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "AnyGraph"))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DATASETS_DIR = os.path.join(BASE_DIR, "Datasets")

def download_models():
    print("\n[1/2] Baixando modelos pré-treinados do AnyGraph (hkuds/AnyGraph)...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    tar_path = hf_hub_download(
        repo_id="hkuds/AnyGraph",
        filename="AnyGraph pretrained models .tar.gz",
        local_dir=MODELS_DIR,
        resume_download=True
    )
    print(f"Arquivo baixado em: {tar_path}")
    print("Extraindo modelos para AnyGraph/Models/...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=MODELS_DIR)
    print("Modelos extraídos com sucesso!")

def download_datasets():
    print("\n[2/2] Baixando datasets zero-shot (hkuds/AnyGraph_datasets)...")
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    zip_path = hf_hub_download(
        repo_id="hkuds/AnyGraph_datasets",
        repo_type="dataset",
        filename="zero-shot datasets.zip",
        local_dir=DATASETS_DIR,
        resume_download=True
    )
    print(f"Arquivo baixado em: {zip_path}")
    print("Extraindo datasets para AnyGraph/Datasets/...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path=DATASETS_DIR)
    print("Datasets extraídos com sucesso!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AnyGraph Models and Datasets")
    parser.add_argument("--models", action="store_true", help="Download pretrained models")
    parser.add_argument("--datasets", action="store_true", help="Download datasets")
    parser.add_argument("--all", action="store_true", help="Download both models and datasets")
    
    args = parser.parse_args()
    
    if not (args.models or args.datasets or args.all):
        parser.print_help()
        print("\nExemplo:")
        print("  python download_assets.py --models")
        print("  python download_assets.py --datasets")
        print("  python download_assets.py --all")
        sys.exit(1)
        
    if args.models or args.all:
        download_models()
    if args.datasets or args.all:
        download_datasets()

"""
Courtesy: Sayak Paul and Chansung Park.
"""

import os
import pandas as pd
from nbformat import reads, NO_CONVERT
from tqdm import tqdm
from datasets import Dataset
from typing import Dict
from huggingface_hub import HfApi, create_repo
import tempfile
import subprocess

MIRROR_DIRECTORY = "hf_public_repos"
DATASET_ID = "hf-stack-v1"
SERIALIZE_IN_CHUNKS = False
FEATHER_FORMAT = "ftr"
PARQUET_FORMAT = "parquet"

# Block the following formats.
IMAGE = ["png", "jpg", "jpeg", "gif"]
VIDEO = ["mp4", "jfif"]
DOC = [
    "key",
    "PDF",
    "pdf",
    "docx",
    "xlsx",
    "pptx",
]
AUDIO = ["flac", "ogg", "mid", "webm", "wav", "mp3"]
ARCHIVE = ["jar", "aar", "gz", "zip", "bz2"]
MODEL = ["onnx", "pickle", "model", "neuron"]
OTHERS = [
    "npy",
    "index",
    "inv",
    "index",
    "DS_Store",
    "rdb",
    "pack",
    "idx",
    "glb",
    "gltf",
    "len",
    "otf",
    "unitypackage",
    "ttf",
    "xz",
    "pcm",
    "opus",
]
ANTI_FOMATS = tuple(IMAGE + VIDEO + DOC + AUDIO + ARCHIVE + OTHERS)


def upload_to_hub(file_format: str, repo_id: str):
    """Moves all the files matching `file_format` to a folder and
    uploads the folder to the Hugging Face Hub."""
    api = HfApi()
    repo_id = create_repo(repo_id=repo_id, exist_ok=True, repo_type="dataset").repo_id

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.makedirs(tmpdirname, exist_ok=True)
        command = f"mv *.{file_format} {tmpdirname}"
        _ = subprocess.run(command.split())
        api.upload_folder(repo_id=repo_id, folder_path=tmpdirname, repo_type="dataset")


def filter_code_cell(cell) -> bool:
    """Filters a code cell w.r.t shell commands, etc."""
    only_shell = cell["source"].startswith("!")
    only_magic = "%%capture" in cell["source"]
    if only_shell or only_magic:
        return False
    else:
        return True


def process_file(directory_name: str, file_path: str) -> Dict[str, str]:
    """Processes a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            if file_path.endswith("ipynb"):
                # Code courtesy: Chansung Park and Sayak Paul.
                code_cell_str = ""
                notebook = reads(content, NO_CONVERT)

                code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code" if filter_code_cell(c)]

                for cell in code_cells:
                    code_cell_str += cell["source"]
                content = code_cell_str
    except Exception:
        content = ""

    return {
        "repo_id": directory_name,
        "file_path": file_path,
        "content": content,
    }


def read_repository_files(directory) -> pd.DataFrame:
    """Reads the files from the locally cloned repositories."""
    file_paths = []
    df = pd.DataFrame(columns=["repo_id", "file_path", "content"])
    chunk_flag = 0

    # Recursively find all files within the directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file_path.endswith(ANTI_FOMATS) and all(
                k not in file_path for k in [".git", "__pycache__", "xcodeproj"]
            ):
                file_paths.append((os.path.dirname(root), file_path))

    # Process files sequentially.
    print(f"Total file paths: {len(file_paths)}.")
    print("Reading file contents...")

    for i, (directory_name, file_path) in enumerate(tqdm(file_paths)):
        file_content = process_file(directory_name, file_path)

        if file_content["content"] != "":
            temp_df = pd.DataFrame.from_dict([file_content])
            df = pd.concat([df, temp_df])

            if SERIALIZE_IN_CHUNKS and len(df) != 0 and (len(df) % SERIALIZE_IN_CHUNKS == 0):
                df_path = f"df_chunk_{chunk_flag}_{len(df)}.{FEATHER_FORMAT}"
                print(f"Serializing dataframe to {df_path}...")
                df.reset_index().to_parquet(df_path)
                del df
                df = pd.DataFrame(columns=["repo_id", "file_path", "content"])
                chunk_flag += 1

    return df


if __name__ == "__main__":
    df = read_repository_files(MIRROR_DIRECTORY)
    print("DataFrame created, creating dataset...")
    upload_to_hub(file_format=PARQUET_FORMAT, repo_id=DATASET_ID)
    print(f"{FEATHER_FORMAT} files uploaded to the Hub.")
    if not SERIALIZE_IN_CHUNKS:
        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(DATASET_ID, private=True)

"""Download the fixed free multilingual encoder used for module acceptance."""
from pathlib import Path
import json

REPO='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
REVISION='e8f8c211226b894fcb81acc59f3b34ba3efd5f42'


def main():
    from huggingface_hub import snapshot_download
    path=snapshot_download(REPO,revision=REVISION,local_dir='data/raw/semantic_encoder',
        allow_patterns=['*.json','*.safetensors','tokenizer*','vocab*','sentencepiece*','special_tokens*','1_Pooling/*','README.md'],max_workers=3)
    destination=Path('reports/reproduction/completion/semantic');destination.mkdir(parents=True,exist_ok=True)
    (destination/'model_source.json').write_text(json.dumps(dict(repo_id=REPO,revision=REVISION,
        local_path=str(path),license='apache-2.0'),indent=2))
    print(f'Downloaded {REPO} at {REVISION} to {path}')


if __name__=='__main__':main()

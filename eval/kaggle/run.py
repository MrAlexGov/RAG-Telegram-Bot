# Kaggle GPU runner для eval/retrieval_eval.py (реранкер на CPU слишком медленный).
import subprocess, shutil
def sh(c):
    print("$", c, flush=True); subprocess.run(c, shell=True, check=True)
sh("git clone -q -b rag-eval https://github.com/MrAlexGov/RAG-Telegram-Bot /tmp/rag")
sh("pip install -q qdrant-client langchain-text-splitters llama-index-core llama-index-vector-stores-qdrant llama-index-embeddings-huggingface")
sh("pip uninstall -y -q torchaudio || true")
sh("cd /tmp/rag && CUDA_VISIBLE_DEVICES=0 python -u eval/retrieval_eval.py --docs 100 --questions 300 --out /kaggle/working")

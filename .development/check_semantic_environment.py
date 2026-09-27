"""Integration smoke check: pinned wheel/model, offline loading and Blackwell CUDA."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / ".inspection_deps"))
sys.path.insert(0, str(root / ".semantic_deps"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
def reject_network(*args, **kwargs):
    raise AssertionError("Model smoke test attempted a network connection")
socket.socket.connect = reject_network
socket.create_connection = reject_network

wheel = root / ".downloads" / "torch-2.10.0+cu130-cp314-cp314-win_amd64.whl"
if wheel.exists():
    digest = hashlib.sha256()
    with wheel.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == "dda35d473dd34cafa0668be176b9ad2cb69b1ff570d0336715a6541e89e27640"
    print("Official PyTorch wheel SHA256 verified", flush=True)
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch", torch.__version__, "; Transformers", transformers.__version__, "; device", device, flush=True)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(), "; capability:", torch.cuda.get_device_capability(), flush=True)
model_dir = root / "models" / "multilingual-e5-small"
manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
assert manifest["revision"] == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModel.from_pretrained(model_dir, local_files_only=True, attn_implementation="eager").to(device).eval()
batch = tokenizer(["query: ремонт телевизора", "passage: Мастер по ремонту телевизоров", "passage: Маникюр и педикюр"],
                  padding=True, truncation=True, max_length=192, return_tensors="pt")
batch = {key: value.to(device) for key, value in batch.items()}
def embed():
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)
a, b = embed(), embed()
assert torch.equal(a, b), "Repeated deterministic encoding changed vectors"
array = a.cpu().numpy()
assert array.shape == (3, 384) and np.isfinite(array).all()
assert np.allclose(np.linalg.norm(array, axis=1), 1, atol=2e-5)
similarity = (a[0:1] @ a[1:].T).cpu().numpy()[0]
assert similarity[0] > similarity[1], "Relevant toy passage scored below unrelated passage"
print("Passed: offline model/tokenizer load, CUDA inference, 384-dim normalization, deterministic encoding, retrieval smoke check", flush=True)
print("Cosines:", similarity.tolist(), flush=True)

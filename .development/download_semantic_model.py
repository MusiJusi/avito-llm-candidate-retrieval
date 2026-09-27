"""One-time setup only: download pinned E5 files; inference is fully offline.

The notebook never invokes this script. Upload the models directory with the
solution, or run this preparation step before disconnecting from the internet.
Files are addressed by immutable revision and verified against a local manifest.
"""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DESTINATION = ROOT / "models" / "multilingual-e5-small"
FILES = ["config.json", "tokenizer_config.json", "special_tokens_map.json",
         "tokenizer.json", "model.safetensors"]
# Official LFS metadata for the immutable revision above, not a checksum of a
# potentially truncated local download.
WEIGHTS_SIZE = 470641600
WEIGHTS_SHA256 = "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"
DESTINATION.mkdir(parents=True, exist_ok=True)
manifest_path = DESTINATION / "manifest.json"
previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None

def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

hashes = {}
for name in FILES:
    path = DESTINATION / name
    is_weights = name == "model.safetensors"
    valid_weights = is_weights and path.exists() and path.stat().st_size == WEIGHTS_SIZE and digest_file(path) == WEIGHTS_SHA256
    if not path.exists() or (is_weights and not valid_weights):
        temporary = path.with_suffix(path.suffix + ".tmp")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        print("Downloading", name, flush=True)
        resume = path.stat().st_size if is_weights and path.exists() and path.stat().st_size < WEIGHTS_SIZE else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={resume}-"} if resume else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            if resume:
                assert response.status == 206 and response.headers["Content-Range"].startswith(f"bytes {resume}-")
                shutil.copyfile(path, temporary)
                print("Resuming at byte", resume, flush=True)
            expected = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
            downloaded = 0
            with temporary.open("ab" if resume else "wb") as output:
                while block := response.read(8 * 1024 * 1024):
                    output.write(block)
                    downloaded += len(block)
            if expected is not None:
                assert downloaded == expected, f"Incomplete HTTP response: {name}, {downloaded}/{expected} bytes"
        if is_weights:
            assert temporary.stat().st_size == WEIGHTS_SIZE, "Truncated E5 weights"
            assert digest_file(temporary) == WEIGHTS_SHA256, "E5 weights do not match official SHA256"
        temporary.replace(path)
    hashes[name] = digest_file(path)
    if is_weights:
        assert hashes[name] == WEIGHTS_SHA256
    elif previous:
        assert hashes[name] == previous["sha256"][name], f"Checksum mismatch: {name}"
    print(name, path.stat().st_size, hashes[name], flush=True)

manifest = {"model_id": MODEL_ID, "revision": REVISION, "license": "MIT", "sha256": hashes}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("Prepared offline model:", DESTINATION)

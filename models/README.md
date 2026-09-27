# Local semantic model

The notebook uses `intfloat/multilingual-e5-small` under the MIT license.
Author/model card: https://huggingface.co/intfloat/multilingual-e5-small
Pinned revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`.

Preparation, once with internet access:

```powershell
python .development/download_semantic_model.py
```

The script puts model weights, the fast tokenizer, configuration and a SHA-256
manifest in `models/multilingual-e5-small/`. Large weights and tokenizer.json are
excluded from Git. Include this entire directory when sharing an offline solution.
The notebook verifies every file and loads with `local_files_only=True`.

No remote API, remote model code or network download is used during inference.
Document/query prefixes, attention-mask mean pooling and L2 normalization follow
the author's model card. We use float32 and deterministic PyTorch operations.

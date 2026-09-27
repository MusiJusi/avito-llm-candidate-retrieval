# Local models

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

## Frozen supervised priors

`retrieval-priors/` contains four small HistGradientBoosting models trained by
this solution on the supplied train data. The evaluation pair supports comparison
and negative mining; the final pair is used only for benchmark predictions.
Their provenance, input data hashes and weight hashes are in `manifest.json`.
The complete original training recipe is preserved in `Avito_v0.3.ipynb`.

The current notebook trains the new LightGBM LambdaRank model. Its final
checkpoint is in `artifacts/ranking-v1/final_ranker_*.joblib`. The offline archive
contains these priors, the new final model and the frozen E5 vectors; no external
service is needed to reproduce the submitted answer.

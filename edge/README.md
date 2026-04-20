# `edge/` — Arabic CST Edge Model

Practical edge model trained on Arabic CST-8K tokens.
Nothing here modifies existing paper experiments in `../training/` or `../data/`.

## How to run (Colab)

1. Open Google Colab → Runtime → T4 GPU
2. Upload two files from `data/tokenized/cst-ar-8k/`:
   - `cst-ar-8k-train-100000.jsonl` (58 MB)
   - `cst-ar-8k-train-100000-vocab.json` (469 KB)
3. Upload `edge/training/colab_edge.py`
4. Run:
   ```
   !pip install transformers onnx onnxruntime -q
   !python colab_edge.py
   ```
5. Download 3 files from `/content/edge_model/`:
   - `model_int8.onnx` — browser model (~8 MB)
   - `vocab.json` — token↔id mappings
   - `model.onnx` — fp32 backup (~25 MB)

Time: ~20 min on T4.

## What's here

```
edge/
  training/
    colab_edge.py           # one file: train + export + quantize
    requirements.txt
  demo/                     # browser demo (to be added)
```

# V7 Compatible Retrieval-Semantic Hybrid

V7 is implemented on `feat/v7-compatible-hybrid` from the packaged V4-A head. Every E0-E4
comparison uses the same content-ID-aligned, strengthened 291-group protocol: seeds 17/29/43,
five outer folds, and three grouped inner folds. Selection and fusion use only outer-training
cross-fits; competition test rows were never available or accessed.

## Gate decision

E4 did not improve the common-fold E0 reference (`0.681964` versus `0.688963`; delta
`-0.006999`, positive in only one of three seeds). The predeclared hard stop therefore applies:
V4-A remains the frozen competition candidate. E1 is the strongest validation ablation at
`0.704575`, but it is not substituted after the E4 gate failure. E2 is retained as a negative
semantic ablation (`0.523985`).

## Reproduction

Create Python 3.12 environment and restore ignored inputs/resources:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-v7-lock.txt
$env:PYTHONPATH = "src"
```

Required ignored inputs are the authenticated official labeled JSON, the official Wikimedia
`bnwiki-20260701-pages-articles` dump extracted to WikiExtractor shards, and the pinned
`csebuetnlp/banglabert` revision `9ce791f330578f50da6bc52b54205166fb5d1c8c`. The model weight
SHA-256 must be `9d33f519f42705d54e65fc1601644a6f4562c3462f96943b32b4184536130f98`.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m olikbochon.v7_e0 --train-data "data\competition\dataset samples.json" --wikipedia-root "data\external\wikimedia\bnwiki-20260701-extracted" --resource-name "Wikimedia bnwiki pages-articles" --resource-version 20260701 --resource-license "CC-BY-SA-4.0 and GFDL-1.3-or-later" --resource-source-url "https://dumps.wikimedia.org/bnwiki/20260701/bnwiki-20260701-pages-articles.xml.bz2" --resource-sha1 fdcf44a8ec36fc8a82b1b7b9f44878c4cb6dcba9
.\.venv\Scripts\python.exe -m olikbochon.v7_e1 --train-data "data\competition\dataset samples.json"
.\.venv\Scripts\python.exe -m olikbochon.v7_retrieval_cache --train-data "data\competition\dataset samples.json" --wikipedia-root "data\external\wikimedia\bnwiki-20260701-extracted" --output "artifacts\v7\retrieval\official_retrieval.joblib" --resource-name "Wikimedia bnwiki pages-articles" --resource-version 20260701 --resource-license "CC-BY-SA-4.0 and GFDL-1.3-or-later" --resource-source-url "https://dumps.wikimedia.org/bnwiki/20260701/bnwiki-20260701-pages-articles.xml.bz2" --resource-sha1 fdcf44a8ec36fc8a82b1b7b9f44878c4cb6dcba9
.\.venv\Scripts\python.exe -m olikbochon.v7_e2 --train-data "data\competition\dataset samples.json" --retrieval-cache "artifacts\v7\retrieval\official_retrieval.joblib" --checkpoint-dir "data\models\banglabert-official-9ce791f" --checkpoint-revision 9ce791f330578f50da6bc52b54205166fb5d1c8c --checkpoint-weight-sha256 9d33f519f42705d54e65fc1601644a6f4562c3462f96943b32b4184536130f98
.\.venv\Scripts\python.exe -m olikbochon.v7_e3
.\.venv\Scripts\python.exe -m olikbochon.v7_e4 --train-data "data\competition\dataset samples.json"
.\.venv\Scripts\python.exe scripts\verify_v7_artifacts.py --train-data "data\competition\dataset samples.json"
```

Inference is implemented but not executed because the competition test file is absent:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe inference\run_v7_inference.py --train-data "data\competition\dataset samples.json" --test-data "D:\path\to\test set.csv" --wikipedia-root "data\external\wikimedia\bnwiki-20260701-extracted"
```

The inference path is offline, preserves input IDs/order, validates uniqueness and dynamic row
count, and emits exactly `id,label`. It does not call an external API.

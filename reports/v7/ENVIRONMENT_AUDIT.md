# V7 Environment Audit

Audit date: 2026-07-19  
OS: Windows 11 (`10.0.26200`)  
System/project Python: 3.12.4  
GPU: none visible (`torch.cuda.is_available() == false`; `nvidia-smi` unavailable)

The ignored project virtual environment contains NumPy 2.0.2, pandas 2.3.3, SciPy 1.15.3,
scikit-learn 1.6.1, joblib 1.5.1, PyTorch 2.7.1, Transformers 4.52.4, tokenizers 0.21.4,
WikiExtractor 3.0.6, and pytest 8.4.1. `requirements-v7-lock.txt` is the resolved environment.
E2 ran on CPU with zero peak VRAM.

Authenticated local resources now present (ignored by Git):

- Wikimedia `bnwiki-20260701-pages-articles.xml.bz2`, 529,372,063 bytes, official SHA-1
  `fdcf44a8ec36fc8a82b1b7b9f44878c4cb6dcba9`, license CC BY-SA 4.0 / GFDL.
- WikiExtractor output: 1,160 files and 438,788 decoded articles. Audited V4 compatibility uses
  AA/AB/AC/AD only: 400 shards and 60,876 usable post-filter articles.
- `csebuetnlp/banglabert` revision `9ce791f330578f50da6bc52b54205166fb5d1c8c`;
  `pytorch_model.bin` is 442,560,329 bytes with SHA-256
  `9d33f519f42705d54e65fc1601644a6f4562c3462f96943b32b4184536130f98`.

The historical V4-A Kaggle summary records Python 3.12.13 with NumPy 2.0.2, pandas 2.3.3,
and scikit-learn 1.6.1. Those values describe the preserved run, not this workstation.

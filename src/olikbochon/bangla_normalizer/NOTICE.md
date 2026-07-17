# Official BanglaBERT normalizer notice

The code in `const.py` and `normalize.py` is vendored from the official
[`csebuetnlp/normalizer`](https://github.com/csebuetnlp/normalizer) repository
at immutable commit `d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9`.

The upstream project describes itself as a Python port of the text
normalization used in:

> Tahmid Hasan, Abhik Bhattacharjee, Kazi Samin, Masum Hasan, Madhusudan Basak,
> M. Sohel Rahman, and Rifat Shahriyar. “Not Low-Resource Anymore: Aligner
> Ensembling, Batch Filtering, and New Datasets for Bengali-English Machine
> Translation.” EMNLP 2020.

Paper: https://aclanthology.org/2020.emnlp-main.207/

The upstream README restricts the repository to noncommercial research under
the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

This vendored copy is used for a noncommercial university student competition.
The implementation files differ only by a provenance/ruff header and removal
of inert trailing spaces; the package `__init__.py` was adapted to use an
explicit export and record the pinned revision. No upstream copyright notice
beyond the README attribution and license declaration was present at the pinned
revision.

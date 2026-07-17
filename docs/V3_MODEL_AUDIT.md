# Version 3 BanglaBERT Model Audit

## Decision

**Status: AUTHENTICATED AND OFFLINE-LOAD VERIFIED — the unusable third-party mirror has been retired.**

The official `csebuetnlp/banglabert` snapshot at immutable revision `9ce791f330578f50da6bc52b54205166fb5d1c8c` was downloaded directly from Hugging Face, strictly verified, hashed, and packaged as a private Kaggle dataset. This resolves the earlier model-origin blocker without trusting or redistributing `reasat/banglabert`.

No model was deserialized or loaded during the packaging-only phase. A later
CPU-only synthetic audit successfully loaded the authenticated tokenizer,
pretraining model, and sequence-classification transition with offline flags,
local-only resolution, remote code disabled, and zero network attempts. See
`docs/V3_OFFLINE_LOAD_AUDIT.md`. No training, notebook implementation, or
competition-test access occurred.

## Authenticated replacement

| Item | Recorded value |
|---|---|
| Download tool environment | `/tmp/olikbochon-hf-download-venv` |
| Python | `3.14.4` |
| `huggingface_hub` and `hf` CLI | `1.23.0` |
| Download timestamp | `2026-07-17T00:08:26+06:00` |
| Local ignored snapshot | `data/models/banglabert-official-9ce791f/` |
| Strict verification | Passed: 7 files checked with missing-file and extra-file failures enabled |
| Official file count | 7 |
| Official total size | 443,099,138 bytes |
| Private Kaggle dataset | `abdullahibnyousuf/banglabert-official-snapshot-9ce791f` |
| Kaggle dataset ID | `11219262` |
| Initial version | 1 |
| Kaggle status | `ready` |
| Kaggle visibility | Private (`isPrivate: true`) |
| Kaggle license metadata | `CC-BY-NC-SA-4.0` |

The first strict verification attempt correctly rejected Hugging Face's generated `.cache/huggingface/` local metadata as extra files. That generated cache was moved outside the snapshot, as required by the packaging decision, and the identical strict command then passed for exactly the seven upstream files. The cache is not counted, manifested, staged, or uploaded.

## Sources and immutable reference

| Item | Recorded value |
|---|---|
| Official repository | `https://huggingface.co/csebuetnlp/banglabert` |
| Immutable official revision | `9ce791f330578f50da6bc52b54205166fb5d1c8c` |
| Revision resolution method | `git ls-remote` for `refs/heads/main`, followed by a metadata-only Git clone with LFS smudging disabled |
| Kaggle mirror | `reasat/banglabert` |
| Kaggle dataset ID | `2896818` |
| Kaggle dataset version | `1` |
| Kaggle version timestamp | `2023-02-14T00:33:03.720Z` |
| Kaggle metadata license | `unknown` |
| Audit and manifest retrieval date | `2026-07-16` |
| Mirror download timestamp | Not applicable: the dataset was not downloaded after the remote manifest failed the required-file gate |

The Kaggle mirror's unknown metadata license does not replace or override the official model's CC BY-NC-SA 4.0 license. The project's use decision and redistribution safeguards are recorded in `docs/V3_LICENSE_DECISION.md`.

## Pinned official manifest

The official Git tree at the pinned revision contains seven entries. The weight entry is a Git LFS pointer; its content digest and logical size are taken from that immutable pointer.

| Official path | Logical size (bytes) | Immutable digest |
|---|---:|---|
| `.gitattributes` | 1,175 | SHA-256 `fa057bb09b78fe6d33af5a01440be5e8c881cd8055de35ee5588ea759cf57bfc` |
| `README.md` | 8,501 | SHA-256 `20ff901fcb556271eaabce5c9d868160e9fb5a52a5e11136c66adc667d710496` |
| `config.json` | 586 | SHA-256 `6cfe6529d65d080b18e4541b0409509088bf10182ca85f31f2e51820e11f571f` |
| `pytorch_model.bin` | 442,560,329 | Git LFS SHA-256 `9d33f519f42705d54e65fc1601644a6f4562c3462f96943b32b4184536130f98` |
| `special_tokens_map.json` | 112 | SHA-256 `303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3` |
| `tokenizer_config.json` | 119 | SHA-256 `0f5d257b81eece9884e4b6524cc1806bb256972f4bab32e875b042ab13d9c358` |
| `vocab.txt` | 528,316 | SHA-256 `3914b2a56901adf9a8ec771a52291a45421bd7bc5c13ed5154335e2d065bf1b6` |

The official configuration declares:

- architecture: `ElectraForPreTraining`;
- model type: `electra`;
- hidden and embedding size: 768;
- 12 encoder layers and 12 attention heads;
- intermediate size: 3,072;
- maximum positions: 512; and
- vocabulary size: 32,000.

The official tokenizer package comprises `vocab.txt`, `tokenizer_config.json`, and `special_tokens_map.json`. The model documentation and license declaration are in `README.md`; the official snapshot does not contain a separate license file.

## Historical unusable-mirror manifest summary

The Kaggle API returned 75 paths with a summed declared logical size of 15,136,675 bytes. The following grouped inventory covers every returned path without opening any file content:

| Path group | Files | Contents reported by path and extension |
|---|---:|---|
| `banglabert/.git/**` | 26 | Git administration data, hooks, refs, logs, index, and packed objects |
| Root repository files | 5 | `.gitignore`, `README.md`, `requirements.txt`, `setup.sh`, and `figs/scores.png` |
| `question_answering/**` | 8 | README, Python/task utilities, evaluation/training shell scripts, and three JSON sample-input datasets |
| `sequence_classification/**` | 28 | README, Python task code, evaluation/training shell scripts, and 24 CSV/JSONL/TSV sample-input datasets |
| `token_classification/**` | 8 | README, Python task code, evaluation/training shell scripts, and four JSONL sample-input datasets |
| **Total** | **75** | **15,136,675 declared bytes** |

The mirror includes unexplained executable Python and shell files relative to the expected model-only package. Those files were not executed.

## Historical required-file and hash comparison

| Required official file | Present in mirror manifest | Hash comparison |
|---|---|---|
| `pytorch_model.bin` | No | Impossible |
| `config.json` | No | Impossible |
| `vocab.txt` | No | Impossible |
| `tokenizer_config.json` | No | Impossible |
| `special_tokens_map.json` | No | Impossible |
| `README.md` model documentation at the package root | No equivalent model-package path | Impossible |

There are no required mirror files to compare against the pinned official digests. A repository README or source-code path is not accepted as a substitute for the corresponding official model-package artifact. The mirror therefore cannot be authenticated against the official immutable snapshot.

## Historical blocker

The approved hard-stop conditions were met for that rejected mirror because:

1. all required model and tokenizer files are absent;
2. file-by-file authentication is impossible;
3. the mirror is structurally a source repository rather than a model snapshot; and
4. it includes executable files that are not part of the seven-file official package.

Accordingly, no pickle-based weight may ever be deserialized from that mirror, and no downstream Version 3 implementation may use it as a model source.

## Resolution and private package

The required remediation is complete. The private Kaggle package contains the seven authenticated upstream files plus one locally authored `SNAPSHOT_INFO.md`:

| Uploaded file | Bytes | Provenance |
|---|---:|---|
| `.gitattributes` | 1,175 | Official upstream snapshot |
| `README.md` | 8,501 | Official upstream snapshot and CC BY-NC-SA 4.0 declaration |
| `config.json` | 586 | Official upstream snapshot |
| `pytorch_model.bin` | 442,560,329 | Official upstream snapshot |
| `special_tokens_map.json` | 112 | Official upstream snapshot |
| `tokenizer_config.json` | 119 | Official upstream snapshot |
| `vocab.txt` | 528,316 | Official upstream snapshot |
| `SNAPSHOT_INFO.md` | 2,104 | Locally authored attribution and provenance information |

`SNAPSHOT_INFO.md` has SHA-256 `400abbaac9d4d496279c8e9d15719526f400f5f53d26d0fd46b5eb6d0fb6067e`. It is not an official Hugging Face artifact. The upstream hashes remain exactly as recorded in the pinned official manifest above.

Kaggle acknowledged successful upload of all eight named files, reports the dataset creation status as `ready`, and returns metadata confirming private visibility and the exact license value. The installed CLI's list-files and redownload calls returned HTTP 403 for this private dataset, so no redundant redownload was performed; this access limitation did not expose or modify the dataset.

Updating or replacing the Kaggle input does not authorize public redistribution of the mirror or model weights.

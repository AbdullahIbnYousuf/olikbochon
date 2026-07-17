# Version 3 Official BanglaBERT Normalizer Audit

## Decision

**Passed.** The minimal official BanglaBERT normalizer is pinned, attributed,
vendored, and behaviorally identical to the pinned upstream implementation on
12 synthetic Bengali/Unicode cases. No competition text was used.

## Source and license

| Item | Recorded value |
|---|---|
| Official repository | `https://github.com/csebuetnlp/normalizer` |
| Immutable revision | `d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9` |
| Upstream version | `0.0.1` |
| License declaration | CC BY-NC-SA 4.0, noncommercial research |
| Paper | Hasan et al., “Not Low-Resource Anymore,” EMNLP 2020 |
| Project use | Noncommercial university student competition |

The pinned upstream tree contains no standalone license file or copyright
header. Its README supplies the license declaration, creator/paper attribution,
and citation. Those notices are preserved in the vendored `NOTICE.md`.

## Minimal vendored source

| File | Upstream SHA-256 | Vendored SHA-256 | Handling |
|---|---|---|---|
| `normalizer/__init__.py` | `80423bc06168dcb75d38591352ecbaf8edf3d75a89ecc25f5f914e93222646ad` | `f7161c92062f7d4e76c1a551ad35ac7783ecc960076240dac606f5150a9f6b7a` | Adapted to an explicit export and revisioned version string |
| `normalizer/const.py` | `55a28f8b0c76d729b486508f2e7c12f64cf6a512e178125c8a5aa4041af30c83` | `ed1e3ceaeb70df8e601bd093f7cbb90a144490c4d3aaea6b7c12e433cebb841e` | Provenance/ruff header added; inert trailing spaces removed |
| `normalizer/normalize.py` | `79ab5ea0ea72ffc7109a6150af2f46c9648dbfa3890595fee24997c5ed0c3fbb` | `a0d53efc31dce8fd02628c8631d2ddc3f1e5b4e5f6a1412202d444832825701c` | Provenance/ruff header added; inert trailing spaces removed |
| `NOTICE.md` | Not upstream | `cf20769c413d8d87635855b26e29b153971153d55376138211133c2831251e51` | Locally authored attribution and license notice |

Vendored destination: `src/olikbochon/bangla_normalizer/`.

Intentionally excluded: upstream Git metadata, `.gitignore`, packaging script,
README duplication, and upstream tests. The project notice retains the needed
source, revision, license, attribution, and citation information.

## Runtime dependencies and defaults

Direct upstream dependencies are recorded in
`requirements-v3-normalizer.txt`:

- `emoji==1.4.2`;
- `ftfy==6.0.3`; and
- `regex==2026.7.10` (a Python 3.14-compatible pin of upstream's unbounded
  `regex` requirement).

`ftfy` uses the existing transitive `wcwidth==0.8.2` in the local test
environment. No network call occurs during normalization.

Default behavior is retained exactly: `unicode_norm="NFKC"`, optional
punctuation/URL/emoji replacement disabled, Unicode normalization applied last,
official character and Unicode replacement tables applied, quote variants
normalized, and whitespace runs collapsed.

## Synthetic parity test

The pinned upstream package and vendored package were imported side by side in
the same environment. Twelve synthetic inputs covered:

- ordinary Bengali;
- tabs, newlines, and repeated whitespace;
- quote variants;
- Bengali danda and composition replacements;
- nonbreaking and zero-width spaces;
- full-width Latin letters and digits (NFKC);
- accented Latin text;
- emoji preservation under defaults; and
- URL preservation under defaults.

All 12 vendored results exactly equaled upstream. A separate assertion confirmed
full-width `Ａ` normalizes to `A`, demonstrating the default NFKC behavior.

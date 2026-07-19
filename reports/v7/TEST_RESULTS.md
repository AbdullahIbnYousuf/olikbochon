# V7 Test Results

## Mandatory V7 suite

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v7_context_routing.py tests/test_v7_group_folds.py tests/test_v7_row_alignment.py tests/test_v7_probability_orientation.py tests/test_v7_no_leakage.py -q
```

Result: **33 passed** in 34.09 seconds. The separate inference schema tests also pass.

## Artifact verifier

`scripts/verify_v7_artifacts.py` verified all E0-E4 OOF assignments and reproduced metrics. It
verified 3,588 cross-fit rows for each of E0/E1/E2 and checksums/reload equality for all 60
persisted models. Result: **COMPLETE**, with `test_rows_accessed: 0`.

## Full repository suite

Result: **183 passed, 3 failed** in 64.58 seconds. The failures are the same V3/V4 generated
notebook and vendor byte-digest synchronization checks reproduced on the untouched packaged
V4-A head `c79a68f` on Windows. No V7 test failed.

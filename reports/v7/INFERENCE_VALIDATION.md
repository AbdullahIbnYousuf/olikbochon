# Inference Validation

Status: **COMPLETE**.

The frozen V4-A fallback entry point accepts CSV or records-oriented JSON, discovers the row
count, requires exactly `id,context,prompt_bn,response_bn`, rejects missing/duplicate IDs, trains
only from official labels, accesses test rows only after thresholds are frozen, preserves ID
order, and uses only local Wikipedia resources. Parser/schema tests pass. The actual competition
test bundle was obtained from Kaggle and validated without displaying or manually inspecting any
test example. It contains 2,516 rows with the required input columns and SHA-256
`db75049956c6fa00e4d9c476716ee34bc4cc17a737f52ada06d2c0f80d567b81`.

Because E4 failed the frozen promotion gate, no V7 candidate was promoted. The checksum-verified,
preserved V4-A test probabilities and submission were promoted unchanged. The final submission
has exactly `id,label`, 2,516 unique IDs in official sample order, binary labels only, and matches
the `probability_label1 >= 0.5` decision rule. Its SHA-256 is
`bdfcd907177fb35537ae33998c6427f65d3ce574d18f82ee3e656bdb4ee2fa19`.
